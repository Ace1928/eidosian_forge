from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@functools.total_ordering
class DummySpecies(Species):
    """A special specie for representing non-traditional elements or species. For
    example, representation of vacancies (charged or otherwise), or special
    sites, etc.

    Attributes:
        oxi_state (int): Oxidation state associated with Species.
        Z (int): DummySpecies is always assigned an atomic number equal to the hash
            number of the symbol. Obviously, it makes no sense whatsoever to use
            the atomic number of a Dummy specie for anything scientific. The purpose
            of this is to ensure that for most use cases, a DummySpecies behaves no
            differently from an Element or Species.
        A (int): Just as for Z, to get a DummySpecies to behave like an Element,
            it needs atomic mass number A (arbitrarily set to twice Z).
        X (float): DummySpecies is always assigned a Pauling electronegativity of 0.
    """

    def __init__(self, symbol: str='X', oxidation_state: float | None=0, spin: float | None=None) -> None:
        """
        Args:
            symbol (str): An assigned symbol for the dummy specie. Strict
                rules are applied to the choice of the symbol. The dummy
                symbol cannot have any part of first two letters that will
                constitute an Element symbol. Otherwise, a composition may
                be parsed wrongly. E.g., "X" is fine, but "Vac" is not
                because Vac contains V, a valid Element.
            oxidation_state (float): Oxidation state for dummy specie. Defaults to 0.
                deprecated and retained purely for backward compatibility.
            spin: Spin associated with Species. Defaults to None.
        """
        symbol = symbol.title()
        for idx in range(1, min(2, len(symbol)) + 1):
            if Element.is_valid_symbol(symbol[:idx]):
                raise ValueError(f'{symbol} contains {symbol[:idx]}, which is a valid element symbol')
        self._symbol = symbol
        self._oxi_state = oxidation_state
        self._spin = spin

    def __getattr__(self, attr):
        raise AttributeError

    def __lt__(self, other):
        """Sets a default sort order for atomic species by Pauling electronegativity,
        followed by oxidation state.
        """
        if self.X != other.X:
            return self.X < other.X
        if self.symbol != other.symbol:
            return self.symbol < other.symbol
        other_oxi = 0 if isinstance(other, Element) else other.oxi_state
        return self.oxi_state < other_oxi

    @property
    def Z(self) -> int:
        """
        Proton number of DummySpecies.

        DummySpecies is always assigned an atomic number equal to the hash of
        the symbol. This is necessary for the DummySpecies object to behave like
        an ElementBase object (which Species inherits from).
        """
        return hash(self.symbol)

    @property
    def A(self) -> int | None:
        """
        Atomic mass number of a DummySpecies.

        To behave like an ElementBase object (from which Species inherits),
        DummySpecies needs an atomic mass number. Consistent with the
        implementation for ElementBase, this can return an int or None (default).
        """
        return None

    @property
    def oxi_state(self) -> float | None:
        """Oxidation state associated with DummySpecies."""
        return self._oxi_state

    @property
    def X(self) -> float:
        """DummySpecies is always assigned a Pauling electronegativity of 0. The effect of
        this is that DummySpecies are always sorted in front of actual Species.
        """
        return 0.0

    @property
    def symbol(self) -> str:
        """Symbol for DummySpecies."""
        return self._symbol

    def __deepcopy__(self, memo):
        return DummySpecies(self.symbol, self._oxi_state)

    @classmethod
    def from_str(cls, species_string: str) -> Self:
        """Returns a Dummy from a string representation.

        Args:
            species_string (str): A string representation of a dummy
                species, e.g., "X2+", "X3+".

        Returns:
            A DummySpecies object.

        Raises:
            ValueError if species_string cannot be interpreted.
        """
        m = re.search('([A-ZAa-z]*)([0-9.]*)([+\\-]*)(.*)', species_string)
        if m:
            sym = m.group(1)
            if m.group(2) == m.group(3) == '':
                oxi = 0.0
            else:
                oxi = 1.0 if m.group(2) == '' else float(m.group(2))
                oxi = -oxi if m.group(3) == '-' else oxi
            properties = {}
            if m.group(4):
                tokens = m.group(4).split('=')
                properties = {tokens[0]: float(tokens[1])}
            return cls(sym, oxi, **properties)
        raise ValueError('Invalid DummySpecies String')

    def as_dict(self) -> dict:
        """MSONable dict representation."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'element': self.symbol, 'oxidation_state': self._oxi_state, 'spin': self._spin}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            DummySpecies
        """
        return cls(dct['element'], dct['oxidation_state'], spin=dct.get('spin'))

    def __repr__(self) -> str:
        return f'DummySpecies {self}'

    def __str__(self) -> str:
        output = self.symbol
        if self.oxi_state is not None:
            output += f'{formula_double_format(abs(self.oxi_state))}{('+' if self.oxi_state >= 0 else '-')}'
        if self._spin is not None:
            spin = self._spin
            output += f',spin={spin!r}'
        return output