import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class EnumeratedValidator(BaseValidator):
    """
    "enumerated": {
        "description": "Enumerated value type. The available values are
                        listed in `values`.",
        "requiredOpts": [
            "values"
        ],
        "otherOpts": [
            "dflt",
            "coerceNumber",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, values, array_ok=False, coerce_number=False, **kwargs):
        super(EnumeratedValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.values = values
        self.array_ok = array_ok
        self.coerce_number = coerce_number
        self.kwargs = kwargs
        self.val_regexs = []
        self.regex_replacements = []
        for v in self.values:
            if v and isinstance(v, str) and (v[0] == '/') and (v[-1] == '/') and (len(v) > 1):
                regex_str = v[1:-1]
                self.val_regexs.append(re.compile(regex_str))
                self.regex_replacements.append(EnumeratedValidator.build_regex_replacement(regex_str))
            else:
                self.val_regexs.append(None)
                self.regex_replacements.append(None)

    def __deepcopy__(self, memodict={}):
        """
        A custom deepcopy method is needed here because compiled regex
        objects don't support deepcopy
        """
        cls = self.__class__
        return cls(self.plotly_name, self.parent_name, values=self.values)

    @staticmethod
    def build_regex_replacement(regex_str):
        match = re.match('\\^(\\w)\\(\\[2\\-9\\]\\|\\[1\\-9\\]\\[0\\-9\\]\\+\\)\\?\\( domain\\)\\?\\$', regex_str)
        if match:
            anchor_char = match.group(1)
            return ('^' + anchor_char + '1$', anchor_char)
        else:
            return None

    def perform_replacemenet(self, v):
        """
        Return v with any applicable regex replacements applied
        """
        if isinstance(v, str):
            for repl_args in self.regex_replacements:
                if repl_args:
                    v = re.sub(repl_args[0], repl_args[1], v)
        return v

    def description(self):
        enum_vals = []
        enum_regexs = []
        for v, regex in zip(self.values, self.val_regexs):
            if regex is not None:
                enum_regexs.append(regex.pattern)
            else:
                enum_vals.append(v)
        desc = "    The '{name}' property is an enumeration that may be specified as:".format(name=self.plotly_name)
        if enum_vals:
            enum_vals_str = '\n'.join(textwrap.wrap(repr(enum_vals), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False))
            desc = desc + '\n      - One of the following enumeration values:\n{enum_vals_str}'.format(enum_vals_str=enum_vals_str)
        if enum_regexs:
            enum_regexs_str = '\n'.join(textwrap.wrap(repr(enum_regexs), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False))
            desc = desc + '\n      - A string that matches one of the following regular expressions:\n{enum_regexs_str}'.format(enum_regexs_str=enum_regexs_str)
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def in_values(self, e):
        """
        Return whether a value matches one of the enumeration options
        """
        is_str = isinstance(e, str)
        for v, regex in zip(self.values, self.val_regexs):
            if is_str and regex:
                in_values = fullmatch(regex, e) is not None
            else:
                in_values = e == v
            if in_values:
                return True
        return False

    def validate_coerce(self, v):
        if v is None:
            pass
        elif self.array_ok and is_array(v):
            v_replaced = [self.perform_replacemenet(v_el) for v_el in v]
            invalid_els = [e for e in v_replaced if not self.in_values(e)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            if is_homogeneous_array(v):
                v = copy_to_readonly_numpy_array(v)
            else:
                v = to_scalar_or_list(v)
        else:
            v = self.perform_replacemenet(v)
            if not self.in_values(v):
                self.raise_invalid_val(v)
        return v