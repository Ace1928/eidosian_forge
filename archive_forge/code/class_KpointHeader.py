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
class KpointHeader(SimpleVaspHeaderParser):
    """Reads nkpts and nbands from the line delimiter.
    Then it also searches for the ibzkpts and kpt_weights"""
    LINE_DELIMITER = 'NKPTS'

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = lines[cursor].strip()
        parts = line.split()
        nkpts = int(parts[3])
        nbands = int(parts[-1])
        results = {'nkpts': nkpts, 'nbands': nbands}
        delim2 = 'k-points in reciprocal lattice and weights'
        for offset, line in enumerate(lines[cursor:], start=0):
            line = line.strip()
            if delim2 in line:
                ibzkpts = np.zeros((nkpts, 3))
                kpt_weights = np.zeros(nkpts)
                for nk in range(nkpts):
                    line = lines[cursor + offset + nk + 1].strip()
                    parts = line.split()
                    ibzkpts[nk] = list(map(float, parts[:3]))
                    kpt_weights[nk] = float(parts[-1])
                results['ibzkpts'] = ibzkpts
                results['kpt_weights'] = kpt_weights
                break
        else:
            raise ParseError('Did not find the K-points in the OUTCAR')
        return results