import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
class FullEvalFormatter(Formatter):
    """A String Formatter that allows evaluation of simple expressions.
    
    Any time a format key is not found in the kwargs,
    it will be tried as an expression in the kwargs namespace.
    
    Note that this version allows slicing using [1:2], so you cannot specify
    a format string. Use :class:`EvalFormatter` to permit format strings.
    
    Examples
    --------
    ::

        In [1]: f = FullEvalFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format('{list(range(5))[2:4]}')
        Out[3]: '[2, 3]'

        In [4]: f.format('{3*2}')
        Out[4]: '6'
    """

    def vformat(self, format_string: str, args, kwargs) -> str:
        result = []
        conversion: Optional[str]
        for literal_text, field_name, format_spec, conversion in self.parse(format_string):
            if literal_text:
                result.append(literal_text)
            if field_name is not None:
                if format_spec:
                    field_name = ':'.join([field_name, format_spec])
                obj = eval(field_name, kwargs)
                obj = self.convert_field(obj, conversion)
                result.append(self.format_field(obj, ''))
        return ''.join(result)