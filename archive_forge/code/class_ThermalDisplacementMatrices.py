from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
class ThermalDisplacementMatrices(MSONable):
    """Class to handle thermal displacement matrices
    This class stores thermal displacement matrices in Ucart format.

    An earlier implementation based on Matlab can be found here:
    https://github.com/JaGeo/MolecularToolbox
    ( J. George, A. Wang, V. L. Deringer, R. Wang, R. Dronskowski, U. Englert, CrystEngComm, 2015, 17, 7414-7422.)
    """

    def __init__(self, thermal_displacement_matrix_cart: ArrayLike[ArrayLike], structure: Structure, temperature: float | None, thermal_displacement_matrix_cif: ArrayLike[ArrayLike]=None) -> None:
        """
        Args:
            thermal_displacement_matrix_cart: 2D numpy array including the thermal_displacement matrix Ucart
                1st dimension atom types, then compressed thermal displacement matrix will follow
                U11, U22, U33, U23, U13, U12 (xx, yy, zz, yz, xz, xy)
                convention similar to "thermal_displacement_matrices.yaml" in phonopy
            structure: A pymatgen Structure object
            temperature: temperature at which thermal displacement matrix was determined
            thermal_displacement_matrix_cif: 2D numpy array including the thermal_displacement matrix Ucif format
                1st dimension atom types, then compressed thermal displacement matrix will follow
                U11, U22, U33, U23, U13, U12 (xx, yy, zz, yz, xz, xy)
                convention similar to "thermal_displacement_matrices.yaml" in phonopy.
        """
        self.thermal_displacement_matrix_cart = np.array(thermal_displacement_matrix_cart)
        self.structure = structure
        self.temperature = temperature
        if thermal_displacement_matrix_cif is not None:
            self.thermal_displacement_matrix_cif = np.array(thermal_displacement_matrix_cif)
        else:
            self.thermal_displacement_matrix_cif = None
        self.thermal_displacement_matrix_cart_matrixform = ThermalDisplacementMatrices.get_full_matrix(self.thermal_displacement_matrix_cart)
        if self.thermal_displacement_matrix_cif is not None:
            self.thermal_displacement_matrix_cif_matrixform = ThermalDisplacementMatrices.get_full_matrix(self.thermal_displacement_matrix_cif)

    @staticmethod
    def get_full_matrix(thermal_displacement: ArrayLike[ArrayLike]) -> np.ndarray[np.ndarray]:
        """Transfers the reduced matrix to the full matrix (order of reduced matrix U11, U22, U33, U23, U13, U12).

        Args:
            thermal_displacement: 2d numpy array, first dimension are the atoms

        Returns:
            3d numpy array including thermal displacements, first dimensions are the atoms
        """
        matrixform = np.zeros((len(thermal_displacement), 3, 3))
        for imat, mat in enumerate(thermal_displacement):
            matrixform[imat][0][0] = mat[0]
            matrixform[imat][1][1] = mat[1]
            matrixform[imat][2][2] = mat[2]
            matrixform[imat][1][2] = mat[3]
            matrixform[imat][2][1] = mat[3]
            matrixform[imat][0][2] = mat[4]
            matrixform[imat][2][0] = mat[4]
            matrixform[imat][0][1] = mat[5]
            matrixform[imat][1][0] = mat[5]
        return matrixform

    @staticmethod
    def get_reduced_matrix(thermal_displacement: ArrayLike[ArrayLike]) -> np.ndarray[np.ndarray]:
        """Transfers the full matrix to reduced matrix (order of reduced matrix U11, U22, U33, U23, U13, U12).

        Args:
            thermal_displacement: 2d numpy array, first dimension are the atoms

        Returns:
            3d numpy array including thermal displacements, first dimensions are the atoms
        """
        reduced_matrix = np.zeros((len(thermal_displacement), 6))
        for imat, mat in enumerate(thermal_displacement):
            reduced_matrix[imat][0] = mat[0][0]
            reduced_matrix[imat][1] = mat[1][1]
            reduced_matrix[imat][2] = mat[2][2]
            reduced_matrix[imat][3] = mat[1][2]
            reduced_matrix[imat][4] = mat[0][2]
            reduced_matrix[imat][5] = mat[0][1]
        return reduced_matrix

    @property
    def Ustar(self) -> np.ndarray:
        """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: Ustar as array. First dimension are the atoms in the structure.
        """
        A = self.structure.lattice.matrix.T
        Ainv = np.linalg.inv(A)
        Ustar = []
        for mat in self.thermal_displacement_matrix_cart_matrixform:
            mat_ustar = np.dot(np.dot(Ainv, mat), Ainv.T)
            Ustar.append(mat_ustar)
        return np.array(Ustar)

    @property
    def Ucif(self) -> np.ndarray:
        """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: Ucif as array. First dimension are the atoms in the structure.
        """
        if self.thermal_displacement_matrix_cif is None:
            A = self.structure.lattice.matrix.T
            N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
            Ninv = np.linalg.inv(N)
            Ucif = []
            Ustar = self.Ustar
            for mat in Ustar:
                mat_cif = np.dot(np.dot(Ninv, mat), Ninv.T)
                Ucif.append(mat_cif)
            return np.array(Ucif)
        return self.thermal_displacement_matrix_cif_matrixform

    @property
    def B(self) -> np.ndarray:
        """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: First dimension are the atoms in the structure.
        """
        B = []
        for mat in self.Ucif:
            mat_B = mat * 8 * np.pi ** 2
            B.append(mat_B)
        return np.array(B)

    @property
    def beta(self) -> list:
        """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: First dimension are the atoms in the structure.
        """
        beta = []
        for mat in self.Ustar:
            mat_beta = mat * 2 * np.pi ** 2
            beta.append(mat_beta)
        return beta

    @property
    def U1U2U3(self) -> list:
        """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: eigenvalues of Ucart. First dimension are the atoms in the structure.
        """
        U1U2U3 = []
        for mat in self.thermal_displacement_matrix_cart_matrixform:
            U1U2U3.append(np.linalg.eig(mat)[0])
        return U1U2U3

    def write_cif(self, filename: str) -> None:
        """Writes a cif including thermal displacements.

        Args:
            filename: name of the cif file
        """
        writer = CifWriter(self.structure)
        writer.write_file(filename)
        with open(filename, mode='a') as file:
            file.write('loop_ \n')
            file.write('_atom_site_aniso_label\n')
            file.write('_atom_site_aniso_U_11\n')
            file.write('_atom_site_aniso_U_22\n')
            file.write('_atom_site_aniso_U_33\n')
            file.write('_atom_site_aniso_U_23\n')
            file.write('_atom_site_aniso_U_13\n')
            file.write('_atom_site_aniso_U_12\n')
            file.write(f'# Additional Data for U_Aniso: {self.temperature}\n')
            for idx, (site, matrix) in enumerate(zip(self.structure, self.Ucif)):
                file.write(f'{site.specie.symbol}{idx} {matrix[0][0]} {matrix[1][1]} {matrix[2][2]} {matrix[1][2]} {matrix[0][2]} {matrix[0][1]}\n')

    @staticmethod
    def _angle_dot(a: ArrayLike, b: ArrayLike) -> float:
        dot_product = np.dot(a, b)
        prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
        divided = dot_product / prod_of_norms
        angle_rad = np.arccos(np.round(divided, 10))
        return np.degrees(angle_rad)

    @due.dcite(Doi('10.1039/C9CE00794F'), description='Angle: A new tool for validating theoretically derived anisotropic displacement parameters with experiment.')
    def compute_directionality_quality_criterion(self, other: ThermalDisplacementMatrices) -> list[dict[str, ArrayLike]]:
        """Will compute directionality of prolate displacement ellipsoids as described in
        https://doi.org/10.1039/C9CE00794F with the earlier implementation: https://github.com/damMroz/Angle/.

        Args:
            other: ThermalDisplacementMatrix
            please make sure that the order of the atoms in both objects that are compared
            is the same. Otherwise, this analysis will deliver wrong results

        Returns:
            will return a list including dicts for each atom that include "vector0"
            (largest principal axes of self object),
             "vector1" (largest principal axes of the other object), "angle" between both axes,
              These vectors can then, for example, be drawn into the structure with VESTA.
              Vectors are given in Cartesian coordinates
        """
        for spec1, spec2 in zip(self.structure.species, other.structure.species):
            if spec1 != spec2:
                raise ValueError('Species in both structures are not the same! Please use structures that are similar to each other')
        structure_match = StructureMatcher()
        if not structure_match.fit(struct1=self.structure, struct2=other.structure):
            raise ValueError('Structures have to be similar')
        results = []
        for self_Ucart, other_Ucart in zip(self.thermal_displacement_matrix_cart_matrixform, other.thermal_displacement_matrix_cart_matrixform):
            result_dict = {}
            invUcart_eig_self, invUcart_eigv_self = np.linalg.eig(np.linalg.inv(self_Ucart))
            invUcart_eig_other, invUcart_eigv_other = np.linalg.eig(np.linalg.inv(other_Ucart))
            argmin_self = np.argmin(invUcart_eig_self)
            vec_self = invUcart_eigv_self.transpose()[argmin_self]
            argmin_other = np.argmin(invUcart_eig_other)
            vec_other = invUcart_eigv_other.transpose()[argmin_other]
            result_dict['angle'] = np.min([self._angle_dot(vec_self, vec_other), self._angle_dot(vec_self, vec_other * -1)])
            result_dict['vector0'] = vec_self
            result_dict['vector1'] = vec_other
            results.append(result_dict)
        return results

    def visualize_directionality_quality_criterion(self, other: ThermalDisplacementMatrices, filename: str | PathLike='visualization.vesta', which_structure: int=0) -> None:
        """Will create a VESTA file for visualization of the directionality criterion.

        Args:
            other: ThermalDisplacementMatrices
            filename:           Filename of the VESTA file
            which_structure:    0 means structure of the self object will be used, 1 means structure of the other
                                object will be used
        """
        result = self.compute_directionality_quality_criterion(other=other)
        matrix_cif = self.thermal_displacement_matrix_cif if self.thermal_displacement_matrix_cif is not None else self.get_reduced_matrix(self.Ucif)
        if which_structure == 0:
            structure = self.structure
        elif which_structure == 1:
            structure = other.structure
        with open(filename, mode='w') as file:
            file.write('#VESTA_FORMAT_VERSION 3.5.4\n \n \n')
            file.write('CRYSTAL\n\n')
            file.write('TITLE\n')
            file.write('Directionality Criterion\n\n')
            file.write('GROUP\n')
            file.write('1 1 P 1\n\n')
            file.write('CELLP\n')
            file.write(f'{structure.lattice.a} {structure.lattice.b} {structure.lattice.c} {structure.lattice.alpha} {structure.lattice.beta} {structure.lattice.gamma}\n')
            file.write('  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n')
            file.write('STRUC\n')
            for isite, site in enumerate(structure, start=1):
                file.write(f'{isite} {site.species_string} {site.species_string}{isite} 1.0000 {site.frac_coords[0]} {site.frac_coords[1]} {site.frac_coords[2]} 1a 1\n')
                file.write(' 0.000000 0.000000 0.000000 0.00\n')
            file.write('  0 0 0 0 0 0 0\n')
            file.write('THERT 0\n')
            file.write('THERM\n')
            counter = 1
            for atom_therm, site in zip(matrix_cif, structure):
                file.write(f'{counter} {site.species_string}{counter} {atom_therm[0]} {atom_therm[1]} {atom_therm[2]} {atom_therm[5]} {atom_therm[4]} {atom_therm[3]}\n')
                counter += 1
            file.write('  0 0 0 0 0 0 0 0\n')
            file.write('VECTR\n')
            vector_count = 1
            site_count = 1
            for vectors in result:
                vector0_x = vectors['vector0'][0]
                vector0_y = vectors['vector0'][1]
                vector0_z = vectors['vector0'][2]
                vector1_x = vectors['vector1'][0]
                vector1_y = vectors['vector1'][1]
                vector1_z = vectors['vector1'][2]
                file.write(f'    {vector_count} {vector0_x} {vector0_y} {vector0_z} 0\n')
                file.write(f'    {site_count} 0 0 0 0\n')
                file.write(' 0 0 0 0 0\n')
                vector_count += 1
                file.write(f'    {vector_count} {vector1_x} {vector1_y} {vector1_z} 0\n')
                file.write(f'    {site_count} 0 0 0 0\n')
                vector_count += 1
                site_count += 1
                file.write(' 0 0 0 0 0\n')
            file.write(' 0 0 0 0 0\n')
            file.write('VECTT\n')
            counter = 1
            for _i in range(len(result)):
                file.write(f'{counter} 0.2 255 0 0 1\n')
                counter += 1
                file.write(f'{counter} 0.2 0 0 255 1\n')
                counter += 1
            file.write(' 0 0 0 0 0\n')

    @property
    def ratio_prolate(self) -> np.ndarray:
        """This will compute ratio between largest and smallest eigenvalue of Ucart."""
        ratios = []
        for us in self.U1U2U3:
            ratios.append(np.max(us) / np.min(us))
        return np.array(ratios)

    @classmethod
    def from_Ucif(cls, thermal_displacement_matrix_cif: ArrayLike[ArrayLike], structure: Structure, temperature: float | None=None) -> Self:
        """Starting from a numpy array, it will convert Ucif values into Ucart values and initialize the class.

        Args:
            thermal_displacement_matrix_cif: np.array,
                first dimension are the atoms,
                then reduced form of thermal displacement matrix will follow
                Order as above: U11, U22, U33, U23, U13, U12
            structure: Structure object
            temperature: float
                Corresponding temperature

        Returns:
            ThermalDisplacementMatrices
        """
        thermal_displacement_matrix_cif_matrix_form = ThermalDisplacementMatrices.get_full_matrix(thermal_displacement_matrix_cif)
        A = structure.lattice.matrix.T
        N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
        Ucart = []
        for mat in thermal_displacement_matrix_cif_matrix_form:
            mat_ustar = np.dot(np.dot(N, mat), N.T)
            mat_ucart = np.dot(np.dot(A, mat_ustar), A.T)
            Ucart.append(mat_ucart)
        thermal_displacement_matrix_cart = ThermalDisplacementMatrices.get_reduced_matrix(np.array(Ucart))
        return cls(thermal_displacement_matrix_cart=thermal_displacement_matrix_cart, thermal_displacement_matrix_cif=thermal_displacement_matrix_cif, structure=structure, temperature=temperature)

    def to_structure_with_site_properties_Ucif(self) -> Structure:
        """Transfers this object into a structure with site properties (Ucif).
        This is useful for sorting the atoms in the structure including site properties.
        E.g., with code like this:
        def sort_order(site):
            return [site.specie.X, site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]]
        new_structure0 = Structure.from_sites(sorted(structure0, key=sort_order)).

        Returns:
            Structure
        """
        site_properties: dict = {'U11_cif': [], 'U22_cif': [], 'U33_cif': [], 'U23_cif': [], 'U13_cif': [], 'U12_cif': []}
        if self.thermal_displacement_matrix_cif is None:
            cif_matrix = self.get_reduced_matrix(self.Ucif)
        else:
            cif_matrix = self.thermal_displacement_matrix_cif
        for atom_ucif in cif_matrix:
            site_properties['U11_cif'].append(atom_ucif[0])
            site_properties['U22_cif'].append(atom_ucif[1])
            site_properties['U33_cif'].append(atom_ucif[2])
            site_properties['U23_cif'].append(atom_ucif[3])
            site_properties['U13_cif'].append(atom_ucif[4])
            site_properties['U12_cif'].append(atom_ucif[5])
        return self.structure.copy(site_properties=site_properties)

    @classmethod
    def from_structure_with_site_properties_Ucif(cls, structure: Structure, temperature: float | None=None) -> Self:
        """Will create this object with the help of a structure with site properties.

        Args:
            structure: Structure object including U11_cif, U22_cif, U33_cif, U23_cif, U13_cif, U12_cif as site
            properties
            temperature: temperature for Ucif data

        Returns:
            ThermalDisplacementMatrices
        """
        Ucif_matrix = []
        for site in structure:
            Ucif_matrix.append([site.properties[f'U{idx}_cif'] for idx in (11, 22, 33, 23, 13, 12)])
        return cls.from_Ucif(Ucif_matrix, structure, temperature=temperature)

    @staticmethod
    def from_cif_P1(filename: str) -> list[ThermalDisplacementMatrices]:
        """Reads a cif with P1 symmetry including positions and ADPs.
        Currently, no check of symmetry is performed as CifParser methods cannot be easily reused.

        Args:
            filename: Filename of the CIF.

        Returns:
            ThermalDisplacementMatrices
        """
        cif = CifFile.from_file(filename)
        thermals = []
        for data in cif.data.values():
            lattice = CifParser.get_lattice_no_exception(data)
            all_coords = []
            all_species = []
            for idx in range(len(data['_atom_site_label'])):
                try:
                    symbol = CifParser(filename)._parse_symbol(data['_atom_site_type_symbol'][idx])
                except KeyError:
                    symbol = CifParser(filename)._parse_symbol(data['_atom_site_label'][idx])
                if not symbol:
                    continue
                all_species.append(symbol)
                x = str2float(data['_atom_site_fract_x'][idx])
                y = str2float(data['_atom_site_fract_y'][idx])
                z = str2float(data['_atom_site_fract_z'][idx])
                all_coords.append([x, y, z])
            thermals_Ucif = [[str2float(data['_atom_site_aniso_U_11'][idx]), str2float(data['_atom_site_aniso_U_22'][idx]), str2float(data['_atom_site_aniso_U_33'][idx]), str2float(data['_atom_site_aniso_U_23'][idx]), str2float(data['_atom_site_aniso_U_13'][idx]), str2float(data['_atom_site_aniso_U_12'][idx])] for idx in range(len(data['_atom_site_aniso_label']))]
            struct = Structure(lattice, all_species, all_coords)
            thermal = ThermalDisplacementMatrices.from_Ucif(thermal_displacement_matrix_cif=thermals_Ucif, structure=struct, temperature=None)
            thermals.append(thermal)
        return thermals