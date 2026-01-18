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
class DefaultParsersContainer:
    """Container for the default OUTCAR parsers.
    Allows for modification of the global default parsers.
    
    Takes in an arbitrary number of parsers. The parsers should be uninitialized,
    as they are created on request.
    """

    def __init__(self, *parsers_cls):
        self._parsers_dct = {}
        for parser in parsers_cls:
            self.add_parser(parser)

    @property
    def parsers_dct(self) -> dict:
        return self._parsers_dct

    def make_parsers(self):
        """Return a copy of the internally stored parsers.
        Parsers are created upon request."""
        return list((parser() for parser in self.parsers_dct.values()))

    def remove_parser(self, name: str):
        """Remove a parser based on the name. The name must match the parser name exactly."""
        self.parsers_dct.pop(name)

    def add_parser(self, parser) -> None:
        """Add a parser"""
        self.parsers_dct[parser.get_name()] = parser