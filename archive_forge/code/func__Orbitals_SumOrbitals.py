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
@classmethod
def _Orbitals_SumOrbitals(cls, dictio, sum_morbs):
    all_orbitals = ['s', 'p', 'd', 'f', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2', 'dz2', 'f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']
    individual_orbs = {'p': ['px', 'py', 'pz'], 'd': ['dxy', 'dyz', 'dxz', 'dx2', 'dz2'], 'f': ['f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']}
    if not isinstance(dictio, dict):
        raise TypeError("The invalid type of 'dictio' was bound. It should be dict type.")
    if len(dictio) == 0:
        raise KeyError("The 'dictio' is empty. We cannot do anything.")
    for elt in dictio:
        if Element.is_valid_symbol(elt):
            if isinstance(dictio[elt], list):
                if len(dictio[elt]) == 0:
                    raise ValueError(f'The dictio[{elt}] is empty. We cannot do anything')
                for orb in dictio[elt]:
                    if not isinstance(orb, str):
                        raise ValueError(f"The invalid format of orbitals is in 'dictio[{elt}]': {orb}. They should be string.")
                    if orb not in all_orbitals:
                        raise ValueError(f"The invalid name of orbital is given in 'dictio[{elt}]'.")
                    if orb in individual_orbs and len(set(dictio[elt]).intersection(individual_orbs[orb])) != 0:
                        raise ValueError(f"The 'dictio[{elt}]' contains orbitals repeated.")
                nelems = Counter(dictio[elt]).values()
                if sum(nelems) > len(nelems):
                    raise ValueError(f'You put in at least two similar orbitals in dictio[{elt}].')
            else:
                raise TypeError(f"The invalid type of value was put into 'dictio[{elt}]'. It should be list type.")
        else:
            raise KeyError(f"The invalid element was put into 'dictio' as a key: {elt}")
    if sum_morbs is None:
        print('You do not want to sum projection over orbitals.')
    elif not isinstance(sum_morbs, dict):
        raise TypeError("The invalid type of 'sum_orbs' was bound. It should be dict or 'None' type.")
    elif len(sum_morbs) == 0:
        raise KeyError("The 'sum_morbs' is empty. We cannot do anything")
    else:
        for elt in sum_morbs:
            if Element.is_valid_symbol(elt):
                if isinstance(sum_morbs[elt], list):
                    for orb in sum_morbs[elt]:
                        if not isinstance(orb, str):
                            raise TypeError(f"The invalid format of orbitals is in 'sum_morbs[{elt}]': {orb}. They should be string.")
                        if orb not in all_orbitals:
                            raise ValueError(f"The invalid name of orbital in 'sum_morbs[{elt}]' is given.")
                        if orb in individual_orbs and len(set(sum_morbs[elt]) & set(individual_orbs[orb])) != 0:
                            raise ValueError(f"The 'sum_morbs[{elt}]' contains orbitals repeated.")
                    nelems = Counter(sum_morbs[elt]).values()
                    if sum(nelems) > len(nelems):
                        raise ValueError(f'You put in at least two similar orbitals in sum_morbs[{elt}].')
                else:
                    raise TypeError(f"The invalid type of value was put into 'sum_morbs[{elt}]'. It should be list type.")
                if elt not in dictio:
                    raise ValueError(f"You cannot sum projection over orbitals of atoms {elt!r} because they are not mentioned in 'dictio'.")
            else:
                raise KeyError(f"The invalid element was put into 'sum_morbs' as a key: {elt}")
    for elt in dictio:
        if len(dictio[elt]) == 1:
            if len(dictio[elt][0]) > 1:
                if elt in sum_morbs:
                    raise ValueError(f'You cannot sum projection over one individual orbital {dictio[elt][0]!r} of {elt!r}.')
            elif sum_morbs is None:
                pass
            elif elt not in sum_morbs:
                print(f'You do not want to sum projection over orbitals of element: {elt}')
            else:
                if len(sum_morbs[elt]) == 0:
                    raise ValueError(f'The empty list is an invalid value for sum_morbs[{elt}].')
                if len(sum_morbs[elt]) > 1:
                    for orb in sum_morbs[elt]:
                        if dictio[elt][0] not in orb:
                            raise ValueError(f"The invalid orbital {orb!r} was put into 'sum_morbs[{elt}]'.")
                else:
                    if orb == 's' or len(orb) > 1:
                        raise ValueError(f'The invalid orbital {orb!r} was put into sum_orbs[{elt!r}].')
                    sum_morbs[elt] = individual_orbs[dictio[elt][0]]
                    dictio[elt] = individual_orbs[dictio[elt][0]]
        else:
            duplicate = copy.deepcopy(dictio[elt])
            for orb in dictio[elt]:
                if orb in individual_orbs:
                    duplicate.remove(orb)
                    duplicate += individual_orbs[orb]
            dictio[elt] = copy.deepcopy(duplicate)
            if sum_morbs is None:
                pass
            elif elt not in sum_morbs:
                print(f'You do not want to sum projection over orbitals of element: {elt}')
            else:
                if len(sum_morbs[elt]) == 0:
                    raise ValueError(f'The empty list is an invalid value for sum_morbs[{elt}].')
                if len(sum_morbs[elt]) == 1:
                    orb = sum_morbs[elt][0]
                    if orb == 's':
                        raise ValueError("We do not sum projection over only 's' orbital of the same type of element.")
                    if orb in individual_orbs:
                        sum_morbs[elt].pop(0)
                        sum_morbs[elt] += individual_orbs[orb]
                    else:
                        raise ValueError(f'You never sum projection over one orbital in sum_morbs[{elt}]')
                else:
                    duplicate = copy.deepcopy(sum_morbs[elt])
                    for orb in sum_morbs[elt]:
                        if orb in individual_orbs:
                            duplicate.remove(orb)
                            duplicate += individual_orbs[orb]
                    sum_morbs[elt] = copy.deepcopy(duplicate)
                for orb in sum_morbs[elt]:
                    if orb not in dictio[elt]:
                        raise ValueError(f'The orbitals of sum_morbs[{elt}] conflict with those of dictio[{elt}].')
    return (dictio, sum_morbs)