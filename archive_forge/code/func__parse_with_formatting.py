import re
from string import Formatter
@staticmethod
def _parse_with_formatting(string, args, kwargs, *, recursion_depth=2, auto_arg_index=0, recursive=False):
    if recursion_depth < 0:
        raise ValueError('Max string recursion exceeded')
    formatter = Formatter()
    parser = AnsiParser()
    for literal_text, field_name, format_spec, conversion in formatter.parse(string):
        parser.feed(literal_text, raw=recursive)
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
            obj, _ = formatter.get_field(field_name, args, kwargs)
            obj = formatter.convert_field(obj, conversion)
            format_spec, auto_arg_index = Colorizer._parse_with_formatting(format_spec, args, kwargs, recursion_depth=recursion_depth - 1, auto_arg_index=auto_arg_index, recursive=True)
            formatted = formatter.format_field(obj, format_spec)
            parser.feed(formatted, raw=True)
    tokens = parser.done()
    if recursive:
        return (AnsiParser.strip(tokens), auto_arg_index)
    return tokens