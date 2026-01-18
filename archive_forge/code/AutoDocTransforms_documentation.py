from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
description is optional. If specified it is used in
        warning messages for the nodes that don't convert to string properly.
        If not specified then no messages are generated.
        