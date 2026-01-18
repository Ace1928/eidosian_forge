from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
class _ObjectVariable(_AbstractVariable):

    def __init__(self, py_db, name, value, register_variable, is_return_value=False, evaluate_name=None, frame=None):
        _AbstractVariable.__init__(self, py_db)
        self.frame = frame
        self.name = name
        self.value = value
        self._register_variable = register_variable
        self._register_variable(self)
        self._is_return_value = is_return_value
        self.evaluate_name = evaluate_name

    @silence_warnings_decorator
    @overrides(_AbstractVariable.get_children_variables)
    def get_children_variables(self, fmt=None, scope=None):
        _type, _type_name, resolver = get_type(self.value)
        children_variables = []
        if resolver is not None:
            if hasattr(resolver, 'get_contents_debug_adapter_protocol'):
                lst = resolver.get_contents_debug_adapter_protocol(self.value, fmt=fmt)
            else:
                dct = resolver.get_dictionary(self.value)
                lst = sorted(dct.items(), key=lambda tup: sorted_attributes_key(tup[0]))
                lst = [(key, value, None) for key, value in lst]
            lst, group_entries = self._group_entries(lst, handle_return_values=False)
            if group_entries:
                lst = group_entries + lst
            parent_evaluate_name = self.evaluate_name
            if parent_evaluate_name:
                for key, val, evaluate_name in lst:
                    if evaluate_name is not None:
                        if callable(evaluate_name):
                            evaluate_name = evaluate_name(parent_evaluate_name)
                        else:
                            evaluate_name = parent_evaluate_name + evaluate_name
                    variable = _ObjectVariable(self.py_db, key, val, self._register_variable, evaluate_name=evaluate_name, frame=self.frame)
                    children_variables.append(variable)
            else:
                for key, val, evaluate_name in lst:
                    variable = _ObjectVariable(self.py_db, key, val, self._register_variable, frame=self.frame)
                    children_variables.append(variable)
        return children_variables

    def change_variable(self, name, value, py_db, fmt=None):
        children_variable = self.get_child_variable_named(name)
        if children_variable is None:
            return None
        var_data = children_variable.get_var_data()
        evaluate_name = var_data.get('evaluateName')
        if not evaluate_name:
            _type, _type_name, container_resolver = get_type(self.value)
            if hasattr(container_resolver, 'change_var_from_name'):
                try:
                    new_value = eval(value)
                except:
                    return None
                new_key = container_resolver.change_var_from_name(self.value, name, new_value)
                if new_key is not None:
                    return _ObjectVariable(self.py_db, new_key, new_value, self._register_variable, evaluate_name=None, frame=self.frame)
                return None
            else:
                return None
        frame = self.frame
        if frame is None:
            return None
        try:
            Exec('%s=%s' % (evaluate_name, value), frame.f_globals, frame.f_locals)
        except:
            return None
        return self.get_child_variable_named(name, fmt=fmt)