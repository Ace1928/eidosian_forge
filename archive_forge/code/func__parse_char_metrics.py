from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def _parse_char_metrics(fh):
    """
    Parse the given filehandle for character metrics information and return
    the information as dicts.

    It is assumed that the file cursor is on the line behind
    'StartCharMetrics'.

    Returns
    -------
    ascii_d : dict
         A mapping "ASCII num of the character" to `.CharMetrics`.
    name_d : dict
         A mapping "character name" to `.CharMetrics`.

    Notes
    -----
    This function is incomplete per the standard, but thus far parses
    all the sample afm files tried.
    """
    required_keys = {'C', 'WX', 'N', 'B'}
    ascii_d = {}
    name_d = {}
    for line in fh:
        line = _to_str(line.rstrip())
        if line.startswith('EndCharMetrics'):
            return (ascii_d, name_d)
        vals = dict((s.strip().split(' ', 1) for s in line.split(';') if s))
        if not required_keys.issubset(vals):
            raise RuntimeError('Bad char metrics line: %s' % line)
        num = _to_int(vals['C'])
        wx = _to_float(vals['WX'])
        name = vals['N']
        bbox = _to_list_of_floats(vals['B'])
        bbox = list(map(int, bbox))
        metrics = CharMetrics(wx, name, bbox)
        if name == 'Euro':
            num = 128
        elif name == 'minus':
            num = ord('−')
        if num != -1:
            ascii_d[num] = metrics
        name_d[name] = metrics
    raise RuntimeError('Bad parse')