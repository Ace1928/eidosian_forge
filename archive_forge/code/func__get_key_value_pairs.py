import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def _get_key_value_pairs(line):
    """Reads a line of a gaussian input file, which contains keywords and options
    separated according to the rules of the route section.

    Parameters
    ----------
    line (string)
        A line of a gaussian input file.

    Returns
    ---------
    params (dict)
        Contains the keywords and options found in the line.
    """
    params = {}
    line = line.strip(' #')
    line = line.split('!')[0]
    match_iterator = re.finditer('\\(([^\\)]+)\\)', line)
    index_ranges = []
    for match in match_iterator:
        index_range = [match.start(0), match.end(0)]
        options = match.group(1)
        keyword_string = line[:match.start(0)]
        keyword_match_iter = [k for k in re.finditer('[^\\,/\\s]+', keyword_string) if k.group() != '=']
        keyword = keyword_match_iter[-1].group().strip(' =')
        index_range[0] = keyword_match_iter[-1].start()
        params.update({keyword.lower(): options.lower()})
        index_ranges.append(index_range)
    index_ranges.reverse()
    for index_range in index_ranges:
        start = index_range[0]
        stop = index_range[1]
        line = line[0:start] + line[stop + 1:]
    line = re.sub('\\s*=\\s*', '=', line)
    line = [x for x in re.split('[\\s,\\/]', line) if x != '']
    for s in line:
        if '=' in s:
            s = s.split('=')
            keyword = s.pop(0)
            options = s.pop(0)
            params.update({keyword.lower(): options.lower()})
        elif len(s) > 0:
            params.update({s.lower(): None})
    return params