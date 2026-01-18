from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
@staticmethod
def _annotate_structure_with_yt(yt, structure: Structure, zpsp):
    volume_idx = charge_idx = None
    for prop in yt['integration']['properties']:
        if prop['label'] == 'Volume':
            volume_idx = prop['id'] - 1
        elif prop['label'] == '$chg_int':
            charge_idx = prop['id'] - 1

    def get_volume_and_charge(nonequiv_idx):
        attractor = yt['integration']['attractors'][nonequiv_idx - 1]
        if attractor['id'] != nonequiv_idx:
            raise ValueError(f'List of attractors may be un-ordered (wanted id={nonequiv_idx}): {attractor}')
        return (attractor['integrals'][volume_idx], attractor['integrals'][charge_idx])
    volumes = []
    charges = []
    charge_transfer = []
    for idx, site in enumerate(yt['structure']['cell_atoms']):
        if not np.allclose(structure[idx].frac_coords, site['fractional_coordinates']):
            raise IndexError(f"Site in structure doesn't seem to match site in YT integration:\n{structure[idx]}\n{site}")
        volume, charge = get_volume_and_charge(site['nonequivalent_id'])
        volumes.append(volume)
        charges.append(charge)
        if zpsp:
            if structure[idx].species_string in zpsp:
                charge_transfer.append(charge - zpsp[structure[idx].species_string])
            else:
                raise ValueError(f'ZPSP argument does not seem compatible with species in structure ({structure[idx].species_string}): {zpsp}')
    structure = structure.copy()
    structure.add_site_property('bader_volume', volumes)
    structure.add_site_property('bader_charge', charges)
    if zpsp:
        if len(charge_transfer) != len(charges):
            warnings.warn(f'Something went wrong calculating charge transfer: {charge_transfer}')
        else:
            structure.add_site_property('bader_charge_transfer', charge_transfer)
    return structure