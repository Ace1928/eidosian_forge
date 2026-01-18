from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class FunctionState(object):

    def __init__(self, owner, names_taken=set(), scope=None):
        self.names_taken = names_taken
        self.owner = owner
        self.scope = scope
        self.error_label = None
        self.label_counter = 0
        self.labels_used = set()
        self.return_label = self.new_label()
        self.new_error_label()
        self.continue_label = None
        self.break_label = None
        self.yield_labels = []
        self.in_try_finally = 0
        self.exc_vars = None
        self.current_except = None
        self.can_trace = False
        self.gil_owned = True
        self.temps_allocated = []
        self.temps_free = {}
        self.temps_used_type = {}
        self.zombie_temps = set()
        self.temp_counter = 0
        self.closure_temps = None
        self.collect_temps_stack = []
        self.should_declare_error_indicator = False
        self.uses_error_indicator = False
        self.error_without_exception = False
        self.needs_refnanny = False

    def validate_exit(self):
        if self.temps_allocated:
            leftovers = self.temps_in_use()
            if leftovers:
                msg = "TEMPGUARD: Temps left over at end of '%s': %s" % (self.scope.name, ', '.join(['%s [%s]' % (name, ctype) for name, ctype, is_pytemp in sorted(leftovers)]))
                raise RuntimeError(msg)

    def new_label(self, name=None):
        n = self.label_counter
        self.label_counter = n + 1
        label = '%s%d' % (Naming.label_prefix, n)
        if name is not None:
            label += '_' + name
        return label

    def new_yield_label(self, expr_type='yield'):
        label = self.new_label('resume_from_%s' % expr_type)
        num_and_label = (len(self.yield_labels) + 1, label)
        self.yield_labels.append(num_and_label)
        return num_and_label

    def new_error_label(self, prefix=''):
        old_err_lbl = self.error_label
        self.error_label = self.new_label(prefix + 'error')
        return old_err_lbl

    def get_loop_labels(self):
        return (self.continue_label, self.break_label)

    def set_loop_labels(self, labels):
        self.continue_label, self.break_label = labels

    def new_loop_labels(self, prefix=''):
        old_labels = self.get_loop_labels()
        self.set_loop_labels((self.new_label(prefix + 'continue'), self.new_label(prefix + 'break')))
        return old_labels

    def get_all_labels(self):
        return (self.continue_label, self.break_label, self.return_label, self.error_label)

    def set_all_labels(self, labels):
        self.continue_label, self.break_label, self.return_label, self.error_label = labels

    def all_new_labels(self):
        old_labels = self.get_all_labels()
        new_labels = []
        for old_label, name in zip(old_labels, ['continue', 'break', 'return', 'error']):
            if old_label:
                new_labels.append(self.new_label(name))
            else:
                new_labels.append(old_label)
        self.set_all_labels(new_labels)
        return old_labels

    def use_label(self, lbl):
        self.labels_used.add(lbl)

    def label_used(self, lbl):
        return lbl in self.labels_used

    def allocate_temp(self, type, manage_ref, static=False, reusable=True):
        """
        Allocates a temporary (which may create a new one or get a previously
        allocated and released one of the same type). Type is simply registered
        and handed back, but will usually be a PyrexType.

        If type.needs_refcounting, manage_ref comes into play. If manage_ref is set to
        True, the temp will be decref-ed on return statements and in exception
        handling clauses. Otherwise the caller has to deal with any reference
        counting of the variable.

        If not type.needs_refcounting, then manage_ref will be ignored, but it
        still has to be passed. It is recommended to pass False by convention
        if it is known that type will never be a reference counted type.

        static=True marks the temporary declaration with "static".
        This is only used when allocating backing store for a module-level
        C array literals.

        if reusable=False, the temp will not be reused after release.

        A C string referring to the variable is returned.
        """
        if type.is_cv_qualified and (not type.is_reference):
            type = type.cv_base_type
        elif type.is_reference and (not type.is_fake_reference):
            type = type.ref_base_type
        elif type.is_cfunction:
            from . import PyrexTypes
            type = PyrexTypes.c_ptr_type(type)
        elif type.is_cpp_class and (not type.is_fake_reference) and self.scope.directives['cpp_locals']:
            self.scope.use_utility_code(UtilityCode.load_cached('OptionalLocals', 'CppSupport.cpp'))
        if not type.needs_refcounting:
            manage_ref = False
        freelist = self.temps_free.get((type, manage_ref))
        if reusable and freelist is not None and freelist[0]:
            result = freelist[0].pop()
            freelist[1].remove(result)
        else:
            while True:
                self.temp_counter += 1
                result = '%s%d' % (Naming.codewriter_temp_prefix, self.temp_counter)
                if result not in self.names_taken:
                    break
            self.temps_allocated.append((result, type, manage_ref, static))
            if not reusable:
                self.zombie_temps.add(result)
        self.temps_used_type[result] = (type, manage_ref)
        if DebugFlags.debug_temp_code_comments:
            self.owner.putln('/* %s allocated (%s)%s */' % (result, type, '' if reusable else ' - zombie'))
        if self.collect_temps_stack:
            self.collect_temps_stack[-1].add((result, type))
        return result

    def release_temp(self, name):
        """
        Releases a temporary so that it can be reused by other code needing
        a temp of the same type.
        """
        type, manage_ref = self.temps_used_type[name]
        freelist = self.temps_free.get((type, manage_ref))
        if freelist is None:
            freelist = ([], set())
            self.temps_free[type, manage_ref] = freelist
        if name in freelist[1]:
            raise RuntimeError('Temp %s freed twice!' % name)
        if name not in self.zombie_temps:
            freelist[0].append(name)
        freelist[1].add(name)
        if DebugFlags.debug_temp_code_comments:
            self.owner.putln('/* %s released %s*/' % (name, ' - zombie' if name in self.zombie_temps else ''))

    def temps_in_use(self):
        """Return a list of (cname,type,manage_ref) tuples of temp names and their type
        that are currently in use.
        """
        used = []
        for name, type, manage_ref, static in self.temps_allocated:
            freelist = self.temps_free.get((type, manage_ref))
            if freelist is None or name not in freelist[1]:
                used.append((name, type, manage_ref and type.needs_refcounting))
        return used

    def temps_holding_reference(self):
        """Return a list of (cname,type) tuples of temp names and their type
        that are currently in use. This includes only temps
        with a reference counted type which owns its reference.
        """
        return [(name, type) for name, type, manage_ref in self.temps_in_use() if manage_ref and type.needs_refcounting]

    def all_managed_temps(self):
        """Return a list of (cname, type) tuples of refcount-managed Python objects.
        """
        return [(cname, type) for cname, type, manage_ref, static in self.temps_allocated if manage_ref]

    def all_free_managed_temps(self):
        """Return a list of (cname, type) tuples of refcount-managed Python
        objects that are not currently in use.  This is used by
        try-except and try-finally blocks to clean up temps in the
        error case.
        """
        return sorted([(cname, type) for (type, manage_ref), freelist in self.temps_free.items() if manage_ref for cname in freelist[0]])

    def start_collecting_temps(self):
        """
        Useful to find out which temps were used in a code block
        """
        self.collect_temps_stack.append(set())

    def stop_collecting_temps(self):
        return self.collect_temps_stack.pop()

    def init_closure_temps(self, scope):
        self.closure_temps = ClosureTempAllocator(scope)