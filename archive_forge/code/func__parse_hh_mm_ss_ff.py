import time as _time
import math as _math
import sys
from operator import index as _index
def _parse_hh_mm_ss_ff(tstr):
    len_str = len(tstr)
    time_comps = [0, 0, 0, 0]
    pos = 0
    for comp in range(0, 3):
        if len_str - pos < 2:
            raise ValueError('Incomplete time component')
        time_comps[comp] = int(tstr[pos:pos + 2])
        pos += 2
        next_char = tstr[pos:pos + 1]
        if comp == 0:
            has_sep = next_char == ':'
        if not next_char or comp >= 2:
            break
        if has_sep and next_char != ':':
            raise ValueError('Invalid time separator: %c' % next_char)
        pos += has_sep
    if pos < len_str:
        if tstr[pos] not in '.,':
            raise ValueError('Invalid microsecond component')
        else:
            pos += 1
            len_remainder = len_str - pos
            if len_remainder >= 6:
                to_parse = 6
            else:
                to_parse = len_remainder
            time_comps[3] = int(tstr[pos:pos + to_parse])
            if to_parse < 6:
                time_comps[3] *= _FRACTION_CORRECTION[to_parse - 1]
            if len_remainder > to_parse and (not all(map(_is_ascii_digit, tstr[pos + to_parse:]))):
                raise ValueError('Non-digit values in unparsed fraction')
    return time_comps