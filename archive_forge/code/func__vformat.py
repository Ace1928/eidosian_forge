import _string
import re as _re
from collections import ChainMap as _ChainMap
def _vformat(self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0):
    if recursion_depth < 0:
        raise ValueError('Max string recursion exceeded')
    result = []
    for literal_text, field_name, format_spec, conversion in self.parse(format_string):
        if literal_text:
            result.append(literal_text)
        if field_name is not None:
            if field_name == '':
                if auto_arg_index is False:
                    raise ValueError('cannot switch from manual field specification to automatic field numbering')
                field_name = str(auto_arg_index)
                auto_arg_index += 1
            elif field_name.isdigit():
                if auto_arg_index:
                    raise ValueError('cannot switch from manual field specification to automatic field numbering')
                auto_arg_index = False
            obj, arg_used = self.get_field(field_name, args, kwargs)
            used_args.add(arg_used)
            obj = self.convert_field(obj, conversion)
            format_spec, auto_arg_index = self._vformat(format_spec, args, kwargs, used_args, recursion_depth - 1, auto_arg_index=auto_arg_index)
            result.append(self.format_field(obj, format_spec))
    return (''.join(result), auto_arg_index)