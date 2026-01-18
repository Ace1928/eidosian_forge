import re
from string import Formatter
@staticmethod
def _parse_without_formatting(string, *, recursion_depth=2, recursive=False):
    if recursion_depth < 0:
        raise ValueError('Max string recursion exceeded')
    formatter = Formatter()
    parser = AnsiParser()
    messages_color_tokens = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(string):
        if literal_text and literal_text[-1] in '{}':
            literal_text += literal_text[-1]
        parser.feed(literal_text, raw=recursive)
        if field_name is not None:
            if field_name == 'message':
                if recursive:
                    messages_color_tokens.append(None)
                else:
                    color_tokens = parser.current_color_tokens()
                    messages_color_tokens.append(color_tokens)
            field = '{%s' % field_name
            if conversion:
                field += '!%s' % conversion
            if format_spec:
                field += ':%s' % format_spec
            field += '}'
            parser.feed(field, raw=True)
            _, color_tokens = Colorizer._parse_without_formatting(format_spec, recursion_depth=recursion_depth - 1, recursive=True)
            messages_color_tokens.extend(color_tokens)
    return (parser.done(), messages_color_tokens)