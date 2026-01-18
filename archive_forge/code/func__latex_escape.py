from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
@staticmethod
def _latex_escape(data, fmt, output_format):
    if output_format != 'latex':
        return data
    if 'replacements' in fmt:
        if isinstance(data, str):
            for repl in sorted(fmt['replacements']):
                data = data.replace(repl, fmt['replacements'][repl])
    return data