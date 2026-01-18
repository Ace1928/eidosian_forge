from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
def _build_symbols(self, results: _RESULT) -> Sequence[str]:
    if 'symbols' in results:
        return results.pop('symbols')
    for required_key in ('ion_types', 'species'):
        if required_key not in results:
            raise ParseError('Did not find required key "{}" in parsed header results.'.format(required_key))
    ion_types = results.pop('ion_types')
    species = results.pop('species')
    if len(ion_types) != len(species):
        raise ParseError('Expected length of ion_types to be same as species, but got ion_types={} and species={}'.format(len(ion_types), len(species)))
    symbols = []
    for n, sym in zip(ion_types, species):
        symbols.extend(n * [sym])
    return symbols