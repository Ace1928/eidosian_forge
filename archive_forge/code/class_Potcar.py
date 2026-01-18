from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class Potcar(list, MSONable):
    """
    Object for reading and writing POTCAR files for calculations. Consists of a
    list of PotcarSingle.
    """
    FUNCTIONAL_CHOICES = tuple(PotcarSingle.functional_dir)

    def __init__(self, symbols: Sequence[str] | None=None, functional: str | None=None, sym_potcar_map: dict[str, str] | None=None) -> None:
        """
        Args:
            symbols (list[str]): Element symbols for POTCAR. This should correspond
                to the symbols used by VASP. E.g., "Mg", "Fe_pv", etc.
            functional (str): Functional used. To know what functional options
                there are, use Potcar.FUNCTIONAL_CHOICES. Note that VASP has
                different versions of the same functional. By default, the old
                PBE functional is used. If you want the newer ones, use PBE_52 or
                PBE_54. Note that if you intend to compare your results with the
                Materials Project, you should use the default setting. You can also
                override the default by setting PMG_DEFAULT_FUNCTIONAL in your
                .pmgrc.yaml.
            sym_potcar_map (dict): Allows a user to specify a specific element
                symbol to raw POTCAR mapping.
        """
        if functional is None:
            functional = SETTINGS.get('PMG_DEFAULT_FUNCTIONAL', 'PBE')
        super().__init__()
        self.functional = functional
        if symbols is not None:
            self.set_symbols(symbols, functional, sym_potcar_map)

    def __iter__(self) -> Iterator[PotcarSingle]:
        """Boilerplate code. Only here to supply type hint so
        `for psingle in Potcar()` is correctly inferred as PotcarSingle
        """
        return super().__iter__()

    def as_dict(self):
        """MSONable dict representation"""
        return {'functional': self.functional, 'symbols': self.symbols, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Potcar
        """
        return Potcar(symbols=dct['symbols'], functional=dct['functional'])

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Reads Potcar from file.

        Args:
            filename: Filename

        Returns:
            Potcar
        """
        with zopen(filename, mode='rt') as file:
            fdata = file.read()
        potcar = cls()
        functionals = []
        for psingle_str in fdata.split('End of Dataset'):
            if (p_strip := psingle_str.strip()):
                psingle = PotcarSingle(p_strip + '\nEnd of Dataset\n')
                potcar.append(psingle)
                functionals.append(psingle.functional)
        if len(set(functionals)) != 1:
            raise ValueError('File contains incompatible functionals!')
        potcar.functional = functionals[0]
        return potcar

    def __str__(self) -> str:
        return '\n'.join((str(potcar).strip('\n') for potcar in self)) + '\n'

    def write_file(self, filename: str) -> None:
        """
        Write Potcar to a file.

        Args:
            filename (str): filename to write to.
        """
        with zopen(filename, mode='wt') as file:
            file.write(str(self))

    @property
    def symbols(self):
        """Get the atomic symbols of all the atoms in the POTCAR file."""
        return [psingle.symbol for psingle in self]

    @symbols.setter
    def symbols(self, symbols):
        self.set_symbols(symbols, functional=self.functional)

    @property
    def spec(self):
        """Get the atomic symbols and hash of all the atoms in the POTCAR file."""
        return [{'symbol': psingle.symbol, 'hash': psingle.md5_computed_file_hash} for psingle in self]

    def set_symbols(self, symbols: Sequence[str], functional: str | None=None, sym_potcar_map: dict[str, str] | None=None):
        """
        Initialize the POTCAR from a set of symbols. Currently, the POTCARs can
        be fetched from a location specified in .pmgrc.yaml. Use pmg config
        to add this setting.

        Args:
            symbols (list[str]): A list of element symbols
            functional (str): The functional to use. If None, the setting
                PMG_DEFAULT_FUNCTIONAL in .pmgrc.yaml is used, or if this is
                not set, it will default to PBE.
            sym_potcar_map (dict): A map of symbol:raw POTCAR string. If
                sym_potcar_map is specified, POTCARs will be generated from
                the given map data rather than the config file location.
        """
        del self[:]
        if sym_potcar_map:
            self.extend((PotcarSingle(sym_potcar_map[el]) for el in symbols))
        else:
            self.extend((PotcarSingle.from_symbol_and_functional(el, functional) for el in symbols))