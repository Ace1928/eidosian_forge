from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def _number_of_subfigures(self, dictio, dictpa, sum_atoms, sum_morbs):
    if not isinstance(dictpa, dict):
        raise TypeError("The invalid type of 'dictpa' was bound. It should be dict type.")
    if len(dictpa) == 0:
        raise KeyError("The 'dictpa' is empty. We cannot do anything.")
    for elt in dictpa:
        if Element.is_valid_symbol(elt):
            if isinstance(dictpa[elt], list):
                if len(dictpa[elt]) == 0:
                    raise ValueError(f'The dictpa[{elt}] is empty. We cannot do anything')
                _sites = self._bs.structure.sites
                indices = []
                for site_idx in range(len(_sites)):
                    if next(iter(_sites[site_idx]._species)) == Element(elt):
                        indices.append(site_idx + 1)
                for number in dictpa[elt]:
                    if isinstance(number, str):
                        if number.lower() == 'all':
                            dictpa[elt] = indices
                            print(f'You want to consider all {elt!r} atoms.')
                            break
                        raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                    if isinstance(number, int):
                        if number not in indices:
                            raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                    else:
                        raise ValueError(f"You put wrong site numbers in 'dictpa[{elt}]': {number}.")
                nelems = Counter(dictpa[elt]).values()
                if sum(nelems) > len(nelems):
                    raise ValueError(f"You put at least two similar site numbers into 'dictpa[{elt}]'.")
            else:
                raise TypeError(f"The invalid type of value was put into 'dictpa[{elt}]'. It should be list type.")
        else:
            raise KeyError(f"The invalid element was put into 'dictpa' as a key: {elt}")
    if len(list(dictio)) != len(list(dictpa)):
        raise KeyError("The number of keys in 'dictio' and 'dictpa' are not the same.")
    for elt in dictio:
        if elt not in dictpa:
            raise KeyError(f'The element {elt!r} is not in both dictpa and dictio.')
    for elt in dictpa:
        if elt not in dictio:
            raise KeyError(f'The element {elt!r} in not in both dictpa and dictio.')
    if sum_atoms is None:
        print('You do not want to sum projection over atoms.')
    elif not isinstance(sum_atoms, dict):
        raise TypeError("The invalid type of 'sum_atoms' was bound. It should be dict type.")
    elif len(sum_atoms) == 0:
        raise KeyError("The 'sum_atoms' is empty. We cannot do anything.")
    else:
        for elt in sum_atoms:
            if Element.is_valid_symbol(elt):
                if isinstance(sum_atoms[elt], list):
                    if len(sum_atoms[elt]) == 0:
                        raise ValueError(f'The sum_atoms[{elt}] is empty. We cannot do anything')
                    _sites = self._bs.structure.sites
                    indices = []
                    for site_idx in range(len(_sites)):
                        if next(iter(_sites[site_idx]._species)) == Element(elt):
                            indices.append(site_idx + 1)
                    for number in sum_atoms[elt]:
                        if isinstance(number, str):
                            if number.lower() == 'all':
                                sum_atoms[elt] = indices
                                print(f'You want to sum projection over all {elt!r} atoms.')
                                break
                            raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                        if isinstance(number, int):
                            if number not in indices:
                                raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                            if number not in dictpa[elt]:
                                raise ValueError(f'You cannot sum projection with atom number {number!r} because it is not mentioned in dicpta[{elt}]')
                        else:
                            raise ValueError(f"You put wrong site numbers in 'sum_atoms[{elt}]'.")
                    nelems = Counter(sum_atoms[elt]).values()
                    if sum(nelems) > len(nelems):
                        raise ValueError(f"You put at least two similar site numbers into 'sum_atoms[{elt}]'.")
                else:
                    raise TypeError(f"The invalid type of value was put into 'sum_atoms[{elt}]'. It should be list type.")
                if elt not in dictpa:
                    raise ValueError(f"You cannot sum projection over atoms {elt!r} because it is not mentioned in 'dictio'.")
            else:
                raise KeyError(f"The invalid element was put into 'sum_atoms' as a key: {elt}")
            if len(sum_atoms[elt]) == 1:
                raise ValueError(f'We do not sum projection over only one atom: {elt}')
    max_number_figs = 0
    decrease = 0
    for elt in dictio:
        max_number_figs += len(dictio[elt]) * len(dictpa[elt])
    if sum_atoms is None and sum_morbs is None:
        number_figs = max_number_figs
    elif sum_atoms is not None and sum_morbs is None:
        for elt in sum_atoms:
            decrease += (len(sum_atoms[elt]) - 1) * len(dictio[elt])
        number_figs = max_number_figs - decrease
    elif sum_atoms is None and sum_morbs is not None:
        for elt in sum_morbs:
            decrease += (len(sum_morbs[elt]) - 1) * len(dictpa[elt])
        number_figs = max_number_figs - decrease
    elif sum_atoms is not None and sum_morbs is not None:
        for elt in sum_atoms:
            decrease += (len(sum_atoms[elt]) - 1) * len(dictio[elt])
        for elt in sum_morbs:
            if elt in sum_atoms:
                decrease += (len(sum_morbs[elt]) - 1) * (len(dictpa[elt]) - len(sum_atoms[elt]) + 1)
            else:
                decrease += (len(sum_morbs[elt]) - 1) * len(dictpa[elt])
        number_figs = max_number_figs - decrease
    else:
        raise ValueError("Invalid format of 'sum_atoms' and 'sum_morbs'.")
    return (dictpa, sum_atoms, number_figs)