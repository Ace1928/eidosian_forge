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
class _AbstractVariable(object):
    name = None
    value = None
    evaluate_name = None

    def __init__(self, py_db):
        assert py_db is not None
        self.py_db = py_db

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def get_variable_reference(self):
        return id(self.value)

    def get_var_data(self, fmt: Optional[dict]=None, context: Optional[str]=None, **safe_repr_custom_attrs):
        """
        :param dict fmt:
            Format expected by the DAP (keys: 'hex': bool, 'rawString': bool)

        :param context:
            This is the context in which the variable is being requested. Valid values:
                "watch",
                "repl",
                "hover",
                "clipboard"
        """
        timer = Timer()
        safe_repr = SafeRepr()
        if fmt is not None:
            safe_repr.convert_to_hex = fmt.get('hex', False)
            safe_repr.raw_value = fmt.get('rawString', False)
        for key, val in safe_repr_custom_attrs.items():
            setattr(safe_repr, key, val)
        type_name, _type_qualifier, _is_exception_on_eval, resolver, value = get_variable_details(self.value, to_string=safe_repr, context=context)
        is_raw_string = type_name in ('str', 'bytes', 'bytearray')
        attributes = []
        if is_raw_string:
            attributes.append('rawString')
        name = self.name
        if self._is_return_value:
            attributes.append('readOnly')
            name = '(return) %s' % (name,)
        elif name in (TOO_LARGE_ATTR, GENERATED_LEN_ATTR_NAME):
            attributes.append('readOnly')
        try:
            if self.value.__class__ == DAPGrouper:
                type_name = ''
        except:
            pass
        var_data = {'name': name, 'value': value, 'type': type_name}
        if self.evaluate_name is not None:
            var_data['evaluateName'] = self.evaluate_name
        if resolver is not None:
            var_data['variablesReference'] = self.get_variable_reference()
        else:
            var_data['variablesReference'] = 0
        if len(attributes) > 0:
            var_data['presentationHint'] = {'attributes': attributes}
        timer.report_if_compute_repr_attr_slow('', name, type_name)
        return var_data

    def get_children_variables(self, fmt=None, scope=None):
        raise NotImplementedError()

    def get_child_variable_named(self, name, fmt=None, scope=None):
        for child_var in self.get_children_variables(fmt=fmt, scope=scope):
            if child_var.get_name() == name:
                return child_var
        return None

    def _group_entries(self, lst, handle_return_values):
        scope_to_grouper = {}
        group_entries = []
        if isinstance(self.value, DAPGrouper):
            new_lst = lst
        else:
            new_lst = []
            get_presentation = self.py_db.variable_presentation.get_presentation
            for attr_name, attr_value, evaluate_name in lst:
                scope = get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values)
                entry = (attr_name, attr_value, evaluate_name)
                if scope:
                    presentation = get_presentation(scope)
                    if presentation == 'hide':
                        continue
                    elif presentation == 'inline':
                        new_lst.append(entry)
                    else:
                        if scope not in scope_to_grouper:
                            grouper = DAPGrouper(scope)
                            scope_to_grouper[scope] = grouper
                        else:
                            grouper = scope_to_grouper[scope]
                        grouper.contents_debug_adapter_protocol.append(entry)
                else:
                    new_lst.append(entry)
            for scope in DAPGrouper.SCOPES_SORTED:
                grouper = scope_to_grouper.get(scope)
                if grouper is not None:
                    group_entries.append((scope, grouper, None))
        return (new_lst, group_entries)