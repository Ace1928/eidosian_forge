import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
class ClassDoc(NumpyDocString):
    extra_public_methods = ['__call__']

    def __init__(self, cls, doc=None, modulename='', func_doc=FunctionDoc, config={}):
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError('Expected a class or None, but got %r' % cls)
        self._cls = cls
        if 'sphinx' in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()
        self.show_inherited_members = config.get('show_inherited_class_members', True)
        if modulename and (not modulename.endswith('.')):
            modulename += '.'
        self._mod = modulename
        if doc is None:
            if cls is None:
                raise ValueError('No class or documentation string given')
            doc = pydoc.getdoc(cls)
        NumpyDocString.__init__(self, doc)
        _members = config.get('members', [])
        if _members is ALL:
            _members = None
        _exclude = config.get('exclude-members', [])
        if config.get('show_class_members', True) and _exclude is not ALL:

            def splitlines_x(s):
                if not s:
                    return []
                else:
                    return s.splitlines()
            for field, items in [('Methods', self.methods), ('Attributes', self.properties)]:
                if not self[field]:
                    doc_list = []
                    for name in sorted(items):
                        if name in _exclude or (_members and name not in _members):
                            continue
                        try:
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(Parameter(name, '', splitlines_x(doc_item)))
                        except AttributeError:
                            pass
                    self[field] = doc_list

    @property
    def methods(self):
        if self._cls is None:
            return []
        return [name for name, func in inspect.getmembers(self._cls) if (not name.startswith('_') or name in self.extra_public_methods) and isinstance(func, Callable) and self._is_show_member(name)]

    @property
    def properties(self):
        if self._cls is None:
            return []
        return [name for name, func in inspect.getmembers(self._cls) if not name.startswith('_') and (func is None or isinstance(func, property) or inspect.isdatadescriptor(func)) and self._is_show_member(name)]

    def _is_show_member(self, name):
        if self.show_inherited_members:
            return True
        if name not in self._cls.__dict__:
            return False
        return True