from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_type(self, type):
    if type is PyrexTypes.py_object_type:
        return None
    elif self.is_format_c:
        code = type.declaration_code('', for_display=1)
        return code
    elif self.is_format_python:
        annotation = None
        if type.is_string:
            annotation = self.current_directives['c_string_type']
        elif type.is_numeric:
            annotation = type.py_type_name()
        if annotation is None:
            code = type.declaration_code('', for_display=1)
            annotation = code.replace(' ', '_').replace('*', 'p')
        return annotation
    return None