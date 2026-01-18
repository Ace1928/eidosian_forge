from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def _decorate_below(self, row_as_string, output_format, **fmt_dict):
    """This really only makes sense for the text and latex output formats.
        """
    dec_below = fmt_dict.get(self.dec_below, None)
    if dec_below is None:
        result = row_as_string
    else:
        output_format = get_output_format(output_format)
        if output_format == 'txt':
            row0len = len(row_as_string)
            dec_len = len(dec_below)
            repeat, addon = divmod(row0len, dec_len)
            result = row_as_string + '\n' + (dec_below * repeat + dec_below[:addon])
        elif output_format == 'latex':
            result = row_as_string + '\n' + dec_below
        else:
            raise ValueError('I cannot decorate a %s header.' % output_format)
    return result