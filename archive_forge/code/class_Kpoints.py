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
class Kpoints(VaspChunkPropertyParser):

    def has_property(self, cursor: _CURSOR, lines: _CHUNK) -> bool:
        line = lines[cursor]
        if 'spin component 1' in line:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    int(parts[-1])
                except ValueError:
                    pass
                else:
                    return True
        return False

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        nkpts = self.get_from_header('nkpts')
        nbands = self.get_from_header('nbands')
        weights = self.get_from_header('kpt_weights')
        spinpol = self.get_from_header('spinpol')
        nspins = 2 if spinpol else 1
        kpts = []
        for spin in range(nspins):
            assert 'spin component' in lines[cursor]
            cursor += 2
            for _ in range(nkpts):
                line = self.get_line(cursor, lines)
                parts = line.strip().split()
                ikpt = int(parts[1]) - 1
                weight = weights[ikpt]
                cursor += 2
                eigenvalues = np.zeros(nbands)
                occupations = np.zeros(nbands)
                for n in range(nbands):
                    parts = lines[cursor].strip().split()
                    eps_n, f_n = map(float, parts[1:])
                    occupations[n] = f_n
                    eigenvalues[n] = eps_n
                    cursor += 1
                kpt = SinglePointKPoint(weight, spin, ikpt, eps_n=eigenvalues, f_n=occupations)
                kpts.append(kpt)
                cursor += 1
        return {'kpts': kpts}