from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
class PhononBandStructure(MSONable):
    """This is the most generic phonon band structure data possible
    it's defined by a list of qpoints + frequencies for each of them.
    Additional information may be given for frequencies at Gamma, where
    non-analytical contribution may be taken into account.
    """

    def __init__(self, qpoints: Sequence[Kpoint], frequencies: ArrayLike, lattice: Lattice, nac_frequencies: Sequence[Sequence] | None=None, eigendisplacements: ArrayLike=None, nac_eigendisplacements: Sequence[Sequence] | None=None, labels_dict: dict | None=None, coords_are_cartesian: bool=False, structure: Structure | None=None) -> None:
        """
        Args:
            qpoints: list of qpoint as numpy arrays, in frac_coords of the
                given lattice by default
            frequencies: list of phonon frequencies in THz as a numpy array with shape
                (3*len(structure), len(qpoints)). The First index of the array
                refers to the band and the second to the index of the qpoint.
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient.
            nac_frequencies: Frequencies with non-analytical contributions at Gamma in THz.
                A list of tuples. The first element of each tuple should be a list
                defining the direction (not necessarily a versor, will be normalized
                internally). The second element containing the 3*len(structure)
                phonon frequencies with non-analytical correction for that direction.
            eigendisplacements: the phonon eigendisplacements associated to the
                frequencies in Cartesian coordinates. A numpy array of complex
                numbers with shape (3*len(structure), len(qpoints), len(structure), 3).
                The first index of the array refers to the band, the second to the index
                of the qpoint, the third to the atom in the structure and the fourth
                to the Cartesian coordinates.
            nac_eigendisplacements: the phonon eigendisplacements associated to the
                non-analytical frequencies in nac_frequencies in Cartesian coordinates.
                A list of tuples. The first element of each tuple should be a list
                defining the direction. The second element containing a numpy array of
                complex numbers with shape (3*len(structure), len(structure), 3).
            labels_dict: (dict[str, Kpoint]): this links a qpoint (in frac coords or
                Cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian (bool): Whether the qpoint coordinates are Cartesian. Defaults to False.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed to calculate element/orbital
                projections of the band structure.
        """
        self.lattice_rec = lattice
        self.qpoints: list[Kpoint] = []
        self.labels_dict = {}
        self.structure = structure
        if eigendisplacements is None:
            eigendisplacements = np.array([])
        self.eigendisplacements = eigendisplacements
        if labels_dict is None:
            labels_dict = {}
        for q_pt in qpoints:
            label = None
            for key in labels_dict:
                if np.linalg.norm(q_pt - np.array(labels_dict[key])) < 0.0001:
                    label = key
                    self.labels_dict[label] = Kpoint(q_pt, lattice, label=label, coords_are_cartesian=coords_are_cartesian)
            self.qpoints += [Kpoint(q_pt, lattice, label=label, coords_are_cartesian=coords_are_cartesian)]
        self.bands = np.asarray(frequencies)
        self.nb_bands = len(self.bands)
        self.nb_qpoints = len(self.qpoints)
        self.nac_frequencies: list[tuple[list[float], np.ndarray]] = []
        self.nac_eigendisplacements: list[tuple[list[float], np.ndarray]] = []
        if nac_frequencies is not None:
            for freq in nac_frequencies:
                self.nac_frequencies.append(([idx / np.linalg.norm(freq[0]) for idx in freq[0]], freq[1]))
        if nac_eigendisplacements is not None:
            for freq in nac_eigendisplacements:
                self.nac_eigendisplacements.append(([idx / np.linalg.norm(freq[0]) for idx in freq[0]], freq[1]))

    def get_gamma_point(self) -> Kpoint | None:
        """Returns the Gamma q-point as a Kpoint object (or None if not found)."""
        for q_point in self.qpoints:
            if np.allclose(q_point.frac_coords, (0, 0, 0)):
                return q_point
        return None

    def min_freq(self) -> tuple[Kpoint, float]:
        """Returns the q-point where the minimum frequency is reached and its value."""
        idx = np.unravel_index(np.argmin(self.bands), self.bands.shape)
        return (self.qpoints[idx[1]], self.bands[idx])

    def max_freq(self) -> tuple[Kpoint, float]:
        """Returns the q-point where the maximum frequency is reached and its value."""
        idx = np.unravel_index(np.argmax(self.bands), self.bands.shape)
        return (self.qpoints[idx[1]], self.bands[idx])

    def width(self, with_imaginary: bool=False) -> float:
        """Returns the difference between the maximum and minimum frequencies anywhere in the
        band structure, not necessarily at identical same q-points. If with_imaginary is False,
        only positive frequencies are considered.
        """
        if with_imaginary:
            return np.max(self.bands) - np.min(self.bands)
        mask_pos = self.bands >= 0
        return self.bands[mask_pos].max() - self.bands[mask_pos].min()

    def has_imaginary_freq(self, tol: float=0.01) -> bool:
        """True if imaginary frequencies are present anywhere in the band structure. Always True if
        has_imaginary_gamma_freq is True.

        Args:
            tol: Tolerance for determining if a frequency is imaginary. Defaults to 0.01.
        """
        return self.min_freq()[1] + tol < 0

    def has_imaginary_gamma_freq(self, tol: float=0.01) -> bool:
        """Checks if there are imaginary modes at the gamma point and all close points.

        Args:
            tol: Tolerance for determining if a frequency is imaginary. Defaults to 0.01.
        """
        close_points = [q_pt for q_pt in self.qpoints if np.linalg.norm(q_pt.frac_coords) < tol]
        for qpoint in close_points:
            idx = self.qpoints.index(qpoint)
            if any((freq < -tol for freq in self.bands[:, idx])):
                return True
        return False

    @property
    def has_nac(self) -> bool:
        """True if nac_frequencies are present (i.e. the band structure has been
        calculated taking into account Born-charge-derived non-analytical corrections at Gamma).
        """
        return len(self.nac_frequencies) > 0

    @property
    def has_eigendisplacements(self) -> bool:
        """True if eigendisplacements are present."""
        return len(self.eigendisplacements) > 0

    def get_nac_frequencies_along_dir(self, direction: Sequence) -> np.ndarray | None:
        """Returns the nac_frequencies for the given direction (not necessarily a versor).
        None if the direction is not present or nac_frequencies has not been calculated.

        Args:
            direction: the direction as a list of 3 elements

        Returns:
            the frequencies as a numpy array o(3*len(structure), len(qpoints)).
            None if not found.
        """
        versor = [idx / np.linalg.norm(direction) for idx in direction]
        for dist, freq in self.nac_frequencies:
            if np.allclose(versor, dist):
                return freq
        return None

    def get_nac_eigendisplacements_along_dir(self, direction) -> np.ndarray | None:
        """Returns the nac_eigendisplacements for the given direction (not necessarily a versor).
        None if the direction is not present or nac_eigendisplacements has not been calculated.

        Args:
            direction: the direction as a list of 3 elements

        Returns:
            the eigendisplacements as a numpy array of complex numbers with shape
            (3*len(structure), len(structure), 3). None if not found.
        """
        versor = [idx / np.linalg.norm(direction) for idx in direction]
        for dist, eigen_disp in self.nac_eigendisplacements:
            if np.allclose(versor, dist):
                return eigen_disp
        return None

    def asr_breaking(self, tol_eigendisplacements: float=1e-05) -> np.ndarray | None:
        """Returns the breaking of the acoustic sum rule for the three acoustic modes,
        if Gamma is present. None otherwise.
        If eigendisplacements are available they are used to determine the acoustic
        modes: selects the bands corresponding  to the eigendisplacements that
        represent to a translation within tol_eigendisplacements. If these are not
        identified or eigendisplacements are missing the first 3 modes will be used
        (indices [0:3]).
        """
        for idx in range(self.nb_qpoints):
            if np.allclose(self.qpoints[idx].frac_coords, (0, 0, 0)):
                if self.has_eigendisplacements:
                    acoustic_modes_index = []
                    for j in range(self.nb_bands):
                        eig = self.eigendisplacements[j][idx]
                        if np.max(np.abs(eig[1:] - eig[:1])) < tol_eigendisplacements:
                            acoustic_modes_index.append(j)
                    if len(acoustic_modes_index) != 3:
                        acoustic_modes_index = [0, 1, 2]
                    return self.bands[acoustic_modes_index, idx]
                return self.bands[:3, idx]
        return None

    def as_dict(self) -> dict[str, Any]:
        """MSONable dict."""
        dct: dict[str, Any] = {'@module': type(self).__module__, '@class': type(self).__name__, 'lattice_rec': self.lattice_rec.as_dict(), 'qpoints': [q_pt.as_dict()['fcoords'] for q_pt in self.qpoints]}
        dct['bands'] = self.bands.tolist()
        dct['labels_dict'] = {}
        for kpoint_letter, kpoint_object in self.labels_dict.items():
            dct['labels_dict'][kpoint_letter] = kpoint_object.as_dict()['fcoords']
        dct['eigendisplacements'] = {'real': np.real(self.eigendisplacements).tolist(), 'imag': np.imag(self.eigendisplacements).tolist()}
        dct['nac_eigendisplacements'] = [(direction, {'real': np.real(e).tolist(), 'imag': np.imag(e).tolist()}) for direction, e in self.nac_eigendisplacements]
        dct['nac_frequencies'] = [(direction, f.tolist()) for direction, f in self.nac_frequencies]
        if self.structure:
            dct['structure'] = self.structure.as_dict()
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """
        Args:
            dct (dict): Dict representation of PhononBandStructure.

        Returns:
            PhononBandStructure
        """
        lattice_rec = Lattice(dct['lattice_rec']['matrix'])
        eigendisplacements = np.array(dct['eigendisplacements']['real']) + np.array(dct['eigendisplacements']['imag']) * 1j
        nac_eigendisplacements = [(direction, np.array(e['real']) + np.array(e['imag']) * 1j) for direction, e in dct['nac_eigendisplacements']]
        nac_frequencies = [(direction, np.array(f)) for direction, f in dct['nac_frequencies']]
        structure = Structure.from_dict(dct['structure']) if 'structure' in dct else None
        return cls(dct['qpoints'], np.array(dct['bands']), lattice_rec, nac_frequencies, eigendisplacements, nac_eigendisplacements, dct['labels_dict'], structure=structure)