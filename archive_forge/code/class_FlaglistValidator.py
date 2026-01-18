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
class FlaglistValidator(BaseValidator):
    """
    "flaglist": {
        "description": "A string representing a combination of flags
                        (order does not matter here). Combine any of the
                        available `flags` with *+*.
                        (e.g. ('lines+markers')). Values in `extras`
                        cannot be combined.",
        "requiredOpts": [
            "flags"
        ],
        "otherOpts": [
            "dflt",
            "extras",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, flags, extras=None, array_ok=False, **kwargs):
        super(FlaglistValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.flags = flags
        self.extras = extras if extras is not None else []
        self.array_ok = array_ok

    def description(self):
        desc = "    The '{plotly_name}' property is a flaglist and may be specified\n    as a string containing:".format(plotly_name=self.plotly_name)
        desc = desc + "\n      - Any combination of {flags} joined with '+' characters\n        (e.g. '{eg_flag}')".format(flags=self.flags, eg_flag='+'.join(self.flags[:2]))
        if self.extras:
            desc = desc + "\n        OR exactly one of {extras} (e.g. '{eg_extra}')".format(extras=self.extras, eg_extra=self.extras[-1])
        if self.array_ok:
            desc = desc + '\n      - A list or array of the above'
        return desc

    def vc_scalar(self, v):
        if isinstance(v, str):
            v = v.strip()
        if v in self.extras:
            return v
        if not isinstance(v, str):
            return None
        split_vals = [e.strip() for e in re.split('[,+]', v)]
        if all((f in self.flags for f in split_vals)):
            return '+'.join(split_vals)
        else:
            return None

    def validate_coerce(self, v):
        if v is None:
            pass
        elif self.array_ok and is_array(v):
            validated_v = [self.vc_scalar(e) for e in v]
            invalid_els = [el for el, validated_el in zip(v, validated_v) if validated_el is None]
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            if is_homogeneous_array(v):
                v = copy_to_readonly_numpy_array(validated_v, kind='U')
            else:
                v = to_scalar_or_list(v)
        else:
            validated_v = self.vc_scalar(v)
            if validated_v is None:
                self.raise_invalid_val(v)
            v = validated_v
        return v