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
def _summarize_keys_for_plot(self, dictio, dictpa, sum_atoms, sum_morbs):
    individual_orbs = {'p': ['px', 'py', 'pz'], 'd': ['dxy', 'dyz', 'dxz', 'dx2', 'dz2'], 'f': ['f_3', 'f_2', 'f_1', 'f0', 'f1', 'f2', 'f3']}

    def number_label(list_numbers):
        list_numbers = sorted(list_numbers)
        divide = [[]]
        divide[0].append(list_numbers[0])
        group = 0
        for idx in range(1, len(list_numbers)):
            if list_numbers[idx] == list_numbers[idx - 1] + 1:
                divide[group].append(list_numbers[idx])
            else:
                group += 1
                divide.append([list_numbers[idx]])
        label = ''
        for elem in divide:
            if len(elem) > 1:
                label += f'{elem[0]}-{elem[-1]},'
            else:
                label += f'{elem[0]},'
        return label[:-1]

    def orbital_label(list_orbitals):
        divide = {}
        for orb in list_orbitals:
            if orb[0] in divide:
                divide[orb[0]].append(orb)
            else:
                divide[orb[0]] = []
                divide[orb[0]].append(orb)
        label = ''
        for elem, orbs in divide.items():
            if elem == 's':
                label += 's,'
            elif len(orbs) == len(individual_orbs[elem]):
                label += elem + ','
            else:
                orb_label = [orb[1:] for orb in orbs]
                label += f'{elem}{str(orb_label).replace('[', '').replace(']', '').replace(', ', '-')},'
        return label[:-1]
    if sum_atoms is None and sum_morbs is None:
        dictio_d = dictio
        dictpa_d = {elt: [str(anum) for anum in dictpa[elt]] for elt in dictpa}
    elif sum_atoms is not None and sum_morbs is None:
        dictio_d = dictio
        dictpa_d = {}
        for elt in dictpa:
            dictpa_d[elt] = []
            if elt in sum_atoms:
                _sites = self._bs.structure.sites
                indices = []
                for site_idx in range(len(_sites)):
                    if next(iter(_sites[site_idx]._species)) == Element(elt):
                        indices.append(site_idx + 1)
                flag_1 = len(set(dictpa[elt]).intersection(indices))
                flag_2 = len(set(sum_atoms[elt]).intersection(indices))
                if flag_1 == len(indices) and flag_2 == len(indices):
                    dictpa_d[elt].append('all')
                else:
                    for anum in dictpa[elt]:
                        if anum not in sum_atoms[elt]:
                            dictpa_d[elt].append(str(anum))
                    label = number_label(sum_atoms[elt])
                    dictpa_d[elt].append(label)
            else:
                for anum in dictpa[elt]:
                    dictpa_d[elt].append(str(anum))
    elif sum_atoms is None and sum_morbs is not None:
        dictio_d = {}
        for elt in dictio:
            dictio_d[elt] = []
            if elt in sum_morbs:
                for morb in dictio[elt]:
                    if morb not in sum_morbs[elt]:
                        dictio_d[elt].append(morb)
                label = orbital_label(sum_morbs[elt])
                dictio_d[elt].append(label)
            else:
                dictio_d[elt] = dictio[elt]
        dictpa_d = {elt: [str(anum) for anum in dictpa[elt]] for elt in dictpa}
    else:
        dictio_d = {}
        for elt in dictio:
            dictio_d[elt] = []
            if elt in sum_morbs:
                for morb in dictio[elt]:
                    if morb not in sum_morbs[elt]:
                        dictio_d[elt].append(morb)
                label = orbital_label(sum_morbs[elt])
                dictio_d[elt].append(label)
            else:
                dictio_d[elt] = dictio[elt]
        dictpa_d = {}
        for elt in dictpa:
            dictpa_d[elt] = []
            if elt in sum_atoms:
                _sites = self._bs.structure.sites
                indices = []
                for site_idx in range(len(_sites)):
                    if next(iter(_sites[site_idx]._species)) == Element(elt):
                        indices.append(site_idx + 1)
                flag_1 = len(set(dictpa[elt]).intersection(indices))
                flag_2 = len(set(sum_atoms[elt]).intersection(indices))
                if flag_1 == len(indices) and flag_2 == len(indices):
                    dictpa_d[elt].append('all')
                else:
                    for anum in dictpa[elt]:
                        if anum not in sum_atoms[elt]:
                            dictpa_d[elt].append(str(anum))
                    label = number_label(sum_atoms[elt])
                    dictpa_d[elt].append(label)
            else:
                for anum in dictpa[elt]:
                    dictpa_d[elt].append(str(anum))
    return (dictio_d, dictpa_d)