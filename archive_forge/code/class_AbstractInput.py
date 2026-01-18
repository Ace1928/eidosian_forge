from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
class AbstractInput(MutableMapping, abc.ABC):
    """Abstract class defining the methods that must be implemented by Input objects."""

    def __delitem__(self, key):
        return self.vars.__delitem__(key)

    def __getitem__(self, key):
        return self.vars.__getitem__(key)

    def __iter__(self):
        return iter(self.vars)

    def __len__(self):
        return len(self.vars)

    def __setitem__(self, key, value):
        self._check_varname(key)
        return self.vars.__setitem__(key, value)

    def __repr__(self):
        return f'<{type(self).__name__} at {id(self)}>'

    def __str__(self):
        return self.to_str()

    def write(self, filepath='run.abi'):
        """Write the input file to file to filepath."""
        dirname = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(dirname, exist_ok=True)
        with open(filepath, mode='w') as file:
            file.write(str(self))

    def deepcopy(self):
        """Deep copy of the input."""
        return copy.deepcopy(self)

    def set_vars(self, *args, **kwargs):
        """
        Set the value of the variables.
        Return dict with the variables added to the input.

        Example:
            input.set_vars(ecut=10, ionmov=3)
        """
        kwargs.update(dict(*args))
        for varname, varvalue in kwargs.items():
            self[varname] = varvalue
        return kwargs

    def set_vars_ifnotin(self, *args, **kwargs):
        """
        Set the value of the variables but only if the variable is not already present.
        Return dict with the variables added to the input.

        Example:
            input.set_vars(ecut=10, ionmov=3)
        """
        kwargs.update(dict(*args))
        added = {}
        for varname, varvalue in kwargs.items():
            if varname not in self:
                self[varname] = varvalue
                added[varname] = varvalue
        return added

    def pop_vars(self, keys):
        """
        Remove the variables listed in keys.
        Return dictionary with the variables that have been removed.
        Unlike remove_vars, no exception is raised if the variables are not in the input.

        Args:
            keys: string or list of strings with variable names.

        Example:
            inp.pop_vars(["ionmov", "optcell", "ntime", "dilatmx"])
        """
        return self.remove_vars(keys, strict=False)

    def remove_vars(self, keys: Sequence[str], strict: bool=True) -> dict[str, InputVariable]:
        """
        Remove the variables listed in keys.
        Return dictionary with the variables that have been removed.

        Args:
            keys: string or list of strings with variable names.
            strict: If True, KeyError is raised if at least one variable is not present.
        """
        if isinstance(keys, str):
            keys = [keys]
        removed = {}
        for key in keys:
            if strict and key not in self:
                raise KeyError(f'key={key!r} not in self:\n {list(self)}')
            if key in self:
                removed[key] = self.pop(key)
        return removed

    @abc.abstractproperty
    def vars(self):
        """Dictionary with the input variables. Used to implement dict-like interface."""

    @abc.abstractmethod
    def _check_varname(self, key):
        """Check if key is a valid name. Raise self.Error if not valid."""

    @abc.abstractmethod
    def to_str(self):
        """Returns a string with the input."""