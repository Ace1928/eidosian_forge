from __future__ import annotations
import warnings
from typing import Any

    Return list of physical_constant keys containing a given string.

    Parameters
    ----------
    sub : str
        Sub-string to search keys for. By default, return all keys.
    disp : bool
        If True, print the keys that are found and return None.
        Otherwise, return the list of keys without printing anything.

    Returns
    -------
    keys : list or None
        If `disp` is False, the list of keys is returned.
        Otherwise, None is returned.

    Examples
    --------
    >>> from scipy.constants import find, physical_constants

    Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?

    >>> find('boltzmann')
    ['Boltzmann constant',
     'Boltzmann constant in Hz/K',
     'Boltzmann constant in eV/K',
     'Boltzmann constant in inverse meter per kelvin',
     'Stefan-Boltzmann constant']

    Get the constant called 'Boltzmann constant in Hz/K':

    >>> physical_constants['Boltzmann constant in Hz/K']
    (20836619120.0, 'Hz K^-1', 0.0)

    Find constants with 'radius' in the key:

    >>> find('radius')
    ['Bohr radius',
     'classical electron radius',
     'deuteron rms charge radius',
     'proton rms charge radius']
    >>> physical_constants['classical electron radius']
    (2.8179403262e-15, 'm', 1.3e-24)

    