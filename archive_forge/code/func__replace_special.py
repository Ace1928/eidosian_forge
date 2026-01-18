import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
def _replace_special(ttype, value, regex, specialttype, replacefunc=lambda x: x):
    last = 0
    for match in regex.finditer(value):
        start, end = (match.start(), match.end())
        if start != last:
            yield (ttype, value[last:start])
        yield (specialttype, replacefunc(value[start:end]))
        last = end
    if last != len(value):
        yield (ttype, value[last:])