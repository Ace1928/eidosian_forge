from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def _inject_pickle_methods(self, node):
    env = self.current_env()
    if node.scope.directives['auto_pickle'] is False:
        return
    auto_pickle_forced = node.scope.directives['auto_pickle'] is True
    all_members = []
    cls = node.entry.type
    cinit = None
    inherited_reduce = None
    while cls is not None:
        all_members.extend((e for e in cls.scope.var_entries if e.name not in ('__weakref__', '__dict__')))
        cinit = cinit or cls.scope.lookup('__cinit__')
        inherited_reduce = inherited_reduce or cls.scope.lookup('__reduce__') or cls.scope.lookup('__reduce_ex__')
        cls = cls.base_type
    all_members.sort(key=lambda e: e.name)
    if inherited_reduce:
        return
    non_py = [e for e in all_members if not e.type.is_pyobject and (not e.type.can_coerce_to_pyobject(env) or not e.type.can_coerce_from_pyobject(env))]
    structs = [e for e in all_members if e.type.is_struct_or_union]
    if cinit or non_py or (structs and (not auto_pickle_forced)):
        if cinit:
            msg = 'no default __reduce__ due to non-trivial __cinit__'
        elif non_py:
            msg = '%s cannot be converted to a Python object for pickling' % ','.join(('self.%s' % e.name for e in non_py))
        else:
            msg = 'Pickling of struct members such as %s must be explicitly requested with @auto_pickle(True)' % ','.join(('self.%s' % e.name for e in structs))
        if auto_pickle_forced:
            error(node.pos, msg)
        pickle_func = TreeFragment(u'\n                def __reduce_cython__(self):\n                    raise TypeError, "%(msg)s"\n                def __setstate_cython__(self, __pyx_state):\n                    raise TypeError, "%(msg)s"\n                ' % {'msg': msg}, level='c_class', pipeline=[NormalizeTree(None)]).substitute({})
        pickle_func.analyse_declarations(node.scope)
        self.visit(pickle_func)
        node.body.stats.append(pickle_func)
    else:
        for e in all_members:
            if not e.type.is_pyobject:
                e.type.create_to_py_utility_code(env)
                e.type.create_from_py_utility_code(env)
        all_members_names = [e.name for e in all_members]
        checksums = _calculate_pickle_checksums(all_members_names)
        unpickle_func_name = '__pyx_unpickle_%s' % node.punycode_class_name
        unpickle_func = TreeFragment(u'\n                def %(unpickle_func_name)s(__pyx_type, long __pyx_checksum, __pyx_state):\n                    cdef object __pyx_PickleError\n                    cdef object __pyx_result\n                    if __pyx_checksum not in %(checksums)s:\n                        from pickle import PickleError as __pyx_PickleError\n                        raise __pyx_PickleError, "Incompatible checksums (0x%%x vs %(checksums)s = (%(members)s))" %% __pyx_checksum\n                    __pyx_result = %(class_name)s.__new__(__pyx_type)\n                    if __pyx_state is not None:\n                        %(unpickle_func_name)s__set_state(<%(class_name)s> __pyx_result, __pyx_state)\n                    return __pyx_result\n\n                cdef %(unpickle_func_name)s__set_state(%(class_name)s __pyx_result, tuple __pyx_state):\n                    %(assignments)s\n                    if len(__pyx_state) > %(num_members)d and hasattr(__pyx_result, \'__dict__\'):\n                        __pyx_result.__dict__.update(__pyx_state[%(num_members)d])\n                ' % {'unpickle_func_name': unpickle_func_name, 'checksums': '(%s)' % ', '.join(checksums), 'members': ', '.join(all_members_names), 'class_name': node.class_name, 'assignments': '; '.join(('__pyx_result.%s = __pyx_state[%s]' % (v, ix) for ix, v in enumerate(all_members_names))), 'num_members': len(all_members_names)}, level='module', pipeline=[NormalizeTree(None)]).substitute({})
        unpickle_func.analyse_declarations(node.entry.scope)
        self.visit(unpickle_func)
        self.extra_module_declarations.append(unpickle_func)
        pickle_func = TreeFragment(u"\n                def __reduce_cython__(self):\n                    cdef tuple state\n                    cdef object _dict\n                    cdef bint use_setstate\n                    state = (%(members)s)\n                    _dict = getattr(self, '__dict__', None)\n                    if _dict is not None:\n                        state += (_dict,)\n                        use_setstate = True\n                    else:\n                        use_setstate = %(any_notnone_members)s\n                    if use_setstate:\n                        return %(unpickle_func_name)s, (type(self), %(checksum)s, None), state\n                    else:\n                        return %(unpickle_func_name)s, (type(self), %(checksum)s, state)\n\n                def __setstate_cython__(self, __pyx_state):\n                    %(unpickle_func_name)s__set_state(self, __pyx_state)\n                " % {'unpickle_func_name': unpickle_func_name, 'checksum': checksums[0], 'members': ', '.join(('self.%s' % v for v in all_members_names)) + (',' if len(all_members_names) == 1 else ''), 'any_notnone_members': ' or '.join(['self.%s is not None' % e.name for e in all_members if e.type.is_pyobject] or ['False'])}, level='c_class', pipeline=[NormalizeTree(None)]).substitute({})
        pickle_func.analyse_declarations(node.scope)
        self.enter_scope(node, node.scope)
        self.visit(pickle_func)
        self.exit_scope()
        node.body.stats.append(pickle_func)