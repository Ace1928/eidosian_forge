import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
def _guess_quote_and_delimiter(self, data, delimiters):
    """
        Looks for text enclosed between two identical quotes
        (the probable quotechar) which are preceded and followed
        by the same character (the probable delimiter).
        For example:
                         ,'some text',
        The quote with the most wins, same with the delimiter.
        If there is no quotechar the delimiter can't be determined
        this way.
        """
    matches = []
    for restr in ('(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?P=delim)', '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)', '(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)', '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)'):
        regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
        matches = regexp.findall(data)
        if matches:
            break
    if not matches:
        return ('', False, None, 0)
    quotes = {}
    delims = {}
    spaces = 0
    groupindex = regexp.groupindex
    for m in matches:
        n = groupindex['quote'] - 1
        key = m[n]
        if key:
            quotes[key] = quotes.get(key, 0) + 1
        try:
            n = groupindex['delim'] - 1
            key = m[n]
        except KeyError:
            continue
        if key and (delimiters is None or key in delimiters):
            delims[key] = delims.get(key, 0) + 1
        try:
            n = groupindex['space'] - 1
        except KeyError:
            continue
        if m[n]:
            spaces += 1
    quotechar = max(quotes, key=quotes.get)
    if delims:
        delim = max(delims, key=delims.get)
        skipinitialspace = delims[delim] == spaces
        if delim == '\n':
            delim = ''
    else:
        delim = ''
        skipinitialspace = 0
    dq_regexp = re.compile('((%(delim)s)|^)\\W*%(quote)s[^%(delim)s\\n]*%(quote)s[^%(delim)s\\n]*%(quote)s\\W*((%(delim)s)|$)' % {'delim': re.escape(delim), 'quote': quotechar}, re.MULTILINE)
    if dq_regexp.search(data):
        doublequote = True
    else:
        doublequote = False
    return (quotechar, doublequote, delim, skipinitialspace)