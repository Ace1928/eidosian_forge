from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
class AbivarAble(abc.ABC):
    """
    An AbivarAble object provides a method to_abivars that returns a dictionary with the abinit variables.
    """

    @abc.abstractmethod
    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""

    def __str__(self):
        return pformat(self.to_abivars(), indent=1, width=80, depth=None)

    def __contains__(self, key):
        return key in self.to_abivars()