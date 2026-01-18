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
class CompoundArrayValidator(BaseValidator):

    def __init__(self, plotly_name, parent_name, data_class_str, data_docs, **kwargs):
        super(CompoundArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.data_class_str = data_class_str
        self._data_class = None
        self.data_docs = data_docs
        self.module_str = CompoundValidator.compute_graph_obj_module_str(self.data_class_str, parent_name)

    def description(self):
        desc = "    The '{plotly_name}' property is a tuple of instances of\n    {class_str} that may be specified as:\n      - A list or tuple of instances of {module_str}.{class_str}\n      - A list or tuple of dicts of string/value properties that\n        will be passed to the {class_str} constructor\n\n        Supported dict properties:\n            {constructor_params_str}".format(plotly_name=self.plotly_name, class_str=self.data_class_str, module_str=self.module_str, constructor_params_str=self.data_docs)
        return desc

    @property
    def data_class(self):
        if self._data_class is None:
            module = import_module(self.module_str)
            self._data_class = getattr(module, self.data_class_str)
        return self._data_class

    def validate_coerce(self, v, skip_invalid=False):
        if v is None:
            v = []
        elif isinstance(v, (list, tuple)):
            res = []
            invalid_els = []
            for v_el in v:
                if isinstance(v_el, self.data_class):
                    res.append(self.data_class(v_el))
                elif isinstance(v_el, dict):
                    res.append(self.data_class(v_el, skip_invalid=skip_invalid))
                elif skip_invalid:
                    res.append(self.data_class())
                else:
                    res.append(None)
                    invalid_els.append(v_el)
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(res)
        elif skip_invalid:
            v = []
        else:
            self.raise_invalid_val(v)
        return v

    def present(self, v):
        return tuple(v)