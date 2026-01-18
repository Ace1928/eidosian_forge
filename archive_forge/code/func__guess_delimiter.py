import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
def _guess_delimiter(self, data, delimiters):
    """
        The delimiter /should/ occur the same number of times on
        each row. However, due to malformed data, it may not. We don't want
        an all or nothing approach, so we allow for small variations in this
        number.
          1) build a table of the frequency of each character on every line.
          2) build a table of frequencies of this frequency (meta-frequency?),
             e.g.  'x occurred 5 times in 10 rows, 6 times in 1000 rows,
             7 times in 2 rows'
          3) use the mode of the meta-frequency to determine the /expected/
             frequency for that character
          4) find out how often the character actually meets that goal
          5) the character that best meets its goal is the delimiter
        For performance reasons, the data is evaluated in chunks, so it can
        try and evaluate the smallest portion of the data possible, evaluating
        additional chunks as necessary.
        """
    data = list(filter(None, data.split('\n')))
    ascii = [chr(c) for c in range(127)]
    chunkLength = min(10, len(data))
    iteration = 0
    charFrequency = {}
    modes = {}
    delims = {}
    start, end = (0, chunkLength)
    while start < len(data):
        iteration += 1
        for line in data[start:end]:
            for char in ascii:
                metaFrequency = charFrequency.get(char, {})
                freq = line.count(char)
                metaFrequency[freq] = metaFrequency.get(freq, 0) + 1
                charFrequency[char] = metaFrequency
        for char in charFrequency.keys():
            items = list(charFrequency[char].items())
            if len(items) == 1 and items[0][0] == 0:
                continue
            if len(items) > 1:
                modes[char] = max(items, key=lambda x: x[1])
                items.remove(modes[char])
                modes[char] = (modes[char][0], modes[char][1] - sum((item[1] for item in items)))
            else:
                modes[char] = items[0]
        modeList = modes.items()
        total = float(min(chunkLength * iteration, len(data)))
        consistency = 1.0
        threshold = 0.9
        while len(delims) == 0 and consistency >= threshold:
            for k, v in modeList:
                if v[0] > 0 and v[1] > 0:
                    if v[1] / total >= consistency and (delimiters is None or k in delimiters):
                        delims[k] = v
            consistency -= 0.01
        if len(delims) == 1:
            delim = list(delims.keys())[0]
            skipinitialspace = data[0].count(delim) == data[0].count('%c ' % delim)
            return (delim, skipinitialspace)
        start = end
        end += chunkLength
    if not delims:
        return ('', 0)
    if len(delims) > 1:
        for d in self.preferred:
            if d in delims.keys():
                skipinitialspace = data[0].count(d) == data[0].count('%c ' % d)
                return (d, skipinitialspace)
    items = [(v, k) for k, v in delims.items()]
    items.sort()
    delim = items[-1][1]
    skipinitialspace = data[0].count(delim) == data[0].count('%c ' % delim)
    return (delim, skipinitialspace)