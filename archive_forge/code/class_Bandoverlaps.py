from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
class Bandoverlaps(MSONable):
    """
    Class to read in bandOverlaps.lobster files. These files are not created during every Lobster run.
    Attributes:
        band_overlaps_dict (dict[Spin, Dict[str, Dict[str, Union[float, np.ndarray]]]]): A dictionary
            containing the band overlap data of the form: {spin: {"kpoint as string": {"maxDeviation":
            float that describes the max deviation, "matrix": 2D array of the size number of bands
            times number of bands including the overlap matrices with}}}.
        max_deviation (list[float]): A list of floats describing the maximal deviation for each problematic kpoint.
    """

    def __init__(self, filename: str='bandOverlaps.lobster', band_overlaps_dict: dict[Any, dict] | None=None, max_deviation: list[float] | None=None):
        """
        Args:
            filename: filename of the "bandOverlaps.lobster" file.
            band_overlaps_dict: A dictionary containing the band overlap data of the form: {spin: {
                    "k_points" : list of k-point array,
                    "max_deviations": list of max deviations associated with each k-point,
                    "matrices": list of the overlap matrices associated with each k-point
                }}.
            max_deviation (list[float]): A list of floats describing the maximal deviation for each problematic k-point.
        """
        self._filename = filename
        self.band_overlaps_dict = {} if band_overlaps_dict is None else band_overlaps_dict
        self.max_deviation = [] if max_deviation is None else max_deviation
        if not self.band_overlaps_dict:
            with zopen(filename, mode='rt') as file:
                contents = file.read().split('\n')
            spin_numbers = [0, 1] if contents[0].split()[-1] == '0' else [1, 2]
            self._filename = filename
            self._read(contents, spin_numbers)

    def _read(self, contents: list, spin_numbers: list):
        """
        Will read in all contents of the file

        Args:
            contents: list of strings
            spin_numbers: list of spin numbers depending on `Lobster` version.
        """
        spin: Spin = Spin.up
        kpoint_array: list = []
        overlaps: list = []
        for line in contents:
            if f'Overlap Matrix (abs) of the orthonormalized projected bands for spin {spin_numbers[0]}' in line:
                spin = Spin.up
            elif f'Overlap Matrix (abs) of the orthonormalized projected bands for spin {spin_numbers[1]}' in line:
                spin = Spin.down
            elif 'k-point' in line:
                kpoint = line.split(' ')
                kpoint_array = []
                for kpointel in kpoint:
                    if kpointel not in ['at', 'k-point', '']:
                        kpoint_array += [float(kpointel)]
            elif 'maxDeviation' in line:
                if spin not in self.band_overlaps_dict:
                    self.band_overlaps_dict[spin] = {}
                if 'k_points' not in self.band_overlaps_dict[spin]:
                    self.band_overlaps_dict[spin]['k_points'] = []
                if 'max_deviations' not in self.band_overlaps_dict[spin]:
                    self.band_overlaps_dict[spin]['max_deviations'] = []
                if 'matrices' not in self.band_overlaps_dict[spin]:
                    self.band_overlaps_dict[spin]['matrices'] = []
                maxdev = line.split(' ')[2]
                self.band_overlaps_dict[spin]['max_deviations'] += [float(maxdev)]
                self.band_overlaps_dict[spin]['k_points'] += [kpoint_array]
                self.max_deviation += [float(maxdev)]
                overlaps = []
            else:
                rows = []
                for el in line.split(' '):
                    if el != '':
                        rows += [float(el)]
                overlaps += [rows]
                if len(overlaps) == len(rows):
                    self.band_overlaps_dict[spin]['matrices'] += [np.matrix(overlaps)]

    def has_good_quality_maxDeviation(self, limit_maxDeviation: float=0.1) -> bool:
        """
        Will check if the maxDeviation from the ideal bandoverlap is smaller or equal to limit_maxDeviation

        Args:
            limit_maxDeviation: limit of the maxDeviation

        Returns:
            Boolean that will give you information about the quality of the projection.
        """
        return all((deviation <= limit_maxDeviation for deviation in self.max_deviation))

    def has_good_quality_check_occupied_bands(self, number_occ_bands_spin_up: int, number_occ_bands_spin_down: int | None=None, spin_polarized: bool=False, limit_deviation: float=0.1) -> bool:
        """
        Will check if the deviation from the ideal bandoverlap of all occupied bands
        is smaller or equal to limit_deviation.

        Args:
            number_occ_bands_spin_up (int): number of occupied bands of spin up
            number_occ_bands_spin_down (int): number of occupied bands of spin down
            spin_polarized (bool): If True, then it was a spin polarized calculation
            limit_deviation (float): limit of the maxDeviation

        Returns:
            Boolean that will give you information about the quality of the projection
        """
        for matrix in self.band_overlaps_dict[Spin.up]['matrices']:
            for iband1, band1 in enumerate(matrix):
                for iband2, band2 in enumerate(band1):
                    if iband1 < number_occ_bands_spin_up and iband2 < number_occ_bands_spin_up:
                        if iband1 == iband2:
                            if abs(band2 - 1.0).all() > limit_deviation:
                                return False
                        elif band2.all() > limit_deviation:
                            return False
        if spin_polarized:
            for matrix in self.band_overlaps_dict[Spin.down]['matrices']:
                for iband1, band1 in enumerate(matrix):
                    for iband2, band2 in enumerate(band1):
                        if number_occ_bands_spin_down is not None:
                            if iband1 < number_occ_bands_spin_down and iband2 < number_occ_bands_spin_down:
                                if iband1 == iband2:
                                    if abs(band2 - 1.0).all() > limit_deviation:
                                        return False
                                elif band2.all() > limit_deviation:
                                    return False
                        else:
                            raise ValueError('number_occ_bands_spin_down has to be specified')
        return True

    @property
    def bandoverlapsdict(self):
        msg = '`bandoverlapsdict` attribute is deprecated. Use `band_overlaps_dict` instead.'
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.band_overlaps_dict