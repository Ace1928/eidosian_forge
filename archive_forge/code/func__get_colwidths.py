from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def _get_colwidths(self, output_format, **fmt_dict):
    """Return list, the calculated widths of each column."""
    output_format = get_output_format(output_format)
    fmt = self.output_formats[output_format].copy()
    fmt.update(fmt_dict)
    ncols = max((len(row) for row in self))
    request = fmt.get('colwidths')
    if request == 0:
        return [0] * ncols
    elif request is None:
        request = [0] * ncols
    elif isinstance(request, int):
        request = [request] * ncols
    elif len(request) < ncols:
        request = [request[i % len(request)] for i in range(ncols)]
    min_widths = []
    for col in zip(*self):
        maxwidth = max((len(c.format(0, output_format, **fmt)) for c in col))
        min_widths.append(maxwidth)
    result = lmap(max, min_widths, request)
    return result