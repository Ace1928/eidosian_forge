import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class _EnumDict(dict):
    """
    Track enum member order and ensure member names are not reused.

    EnumType will use the names found in self._member_names as the
    enumeration member names.
    """

    def __init__(self):
        super().__init__()
        self._member_names = {}
        self._last_values = []
        self._ignore = []
        self._auto_called = False

    def __setitem__(self, key, value):
        """
        Changes anything not dundered or not a descriptor.

        If an enum member name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.
        """
        if _is_internal_class(self._cls_name, value):
            import warnings
            warnings.warn('In 3.13 classes created inside an enum will not become a member.  Use the `member` decorator to keep the current behavior.', DeprecationWarning, stacklevel=2)
        if _is_private(self._cls_name, key):
            pass
        elif _is_sunder(key):
            if key not in ('_order_', '_generate_next_value_', '_numeric_repr_', '_missing_', '_ignore_', '_iter_member_', '_iter_member_by_value_', '_iter_member_by_def_'):
                raise ValueError('_sunder_ names, such as %r, are reserved for future Enum use' % (key,))
            if key == '_generate_next_value_':
                if self._auto_called:
                    raise TypeError('_generate_next_value_ must be defined before members')
                _gnv = value.__func__ if isinstance(value, staticmethod) else value
                setattr(self, '_generate_next_value', _gnv)
            elif key == '_ignore_':
                if isinstance(value, str):
                    value = value.replace(',', ' ').split()
                else:
                    value = list(value)
                self._ignore = value
                already = set(value) & set(self._member_names)
                if already:
                    raise ValueError('_ignore_ cannot specify already set names: %r' % (already,))
        elif _is_dunder(key):
            if key == '__order__':
                key = '_order_'
        elif key in self._member_names:
            raise TypeError('%r already defined as %r' % (key, self[key]))
        elif key in self._ignore:
            pass
        elif isinstance(value, nonmember):
            value = value.value
        elif _is_descriptor(value):
            pass
        else:
            if key in self:
                raise TypeError('%r already defined as %r' % (key, self[key]))
            elif isinstance(value, member):
                value = value.value
            non_auto_store = True
            single = False
            if isinstance(value, auto):
                single = True
                value = (value,)
            if type(value) is tuple and any((isinstance(v, auto) for v in value)):
                auto_valued = []
                for v in value:
                    if isinstance(v, auto):
                        non_auto_store = False
                        if v.value == _auto_null:
                            v.value = self._generate_next_value(key, 1, len(self._member_names), self._last_values[:])
                            self._auto_called = True
                        v = v.value
                        self._last_values.append(v)
                    auto_valued.append(v)
                if single:
                    value = auto_valued[0]
                else:
                    value = tuple(auto_valued)
            self._member_names[key] = None
            if non_auto_store:
                self._last_values.append(value)
        super().__setitem__(key, value)

    def update(self, members, **more_members):
        try:
            for name in members.keys():
                self[name] = members[name]
        except AttributeError:
            for name, value in members:
                self[name] = value
        for name, value in more_members.items():
            self[name] = value