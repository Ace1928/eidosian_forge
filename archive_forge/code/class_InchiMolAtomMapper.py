from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
class InchiMolAtomMapper(AbstractMolAtomMapper):
    """Pair atoms by inchi labels."""

    def __init__(self, angle_tolerance=10.0):
        """
        Args:
            angle_tolerance (float): Angle threshold to assume linear molecule. In degrees.
        """
        self._angle_tolerance = angle_tolerance
        self._assistant_mapper = IsomorphismMolAtomMapper()

    def as_dict(self):
        """
        Returns:
            MSONable dict.
        """
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'angle_tolerance': self._angle_tolerance}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict Representation.

        Returns:
            InchiMolAtomMapper
        """
        return cls(angle_tolerance=dct['angle_tolerance'])

    @staticmethod
    def _inchi_labels(mol):
        """
        Get the inchi canonical labels of the heavy atoms in the molecule.

        Args:
            mol: The molecule. OpenBabel OBMol object

        Returns:
            The label mappings. List of tuple of canonical label,
            original label
            List of equivalent atoms.
        """
        ob_conv = openbabel.OBConversion()
        ob_conv.SetOutFormat('inchi')
        ob_conv.AddOption('a', openbabel.OBConversion.OUTOPTIONS)
        ob_conv.AddOption('X', openbabel.OBConversion.OUTOPTIONS, 'DoNotAddH')
        inchi_text = ob_conv.WriteString(mol)
        match = re.search('InChI=(?P<inchi>.+)\\nAuxInfo=.+/N:(?P<labels>[0-9,;]+)/(E:(?P<eq_atoms>[0-9,;\\(\\)]*)/)?', inchi_text)
        inchi = match.group('inchi')
        label_text = match.group('labels')
        eq_atom_text = match.group('eq_atoms')
        heavy_atom_labels = tuple((int(idx) for idx in label_text.replace(';', ',').split(',')))
        eq_atoms = []
        if eq_atom_text is not None:
            eq_tokens = re.findall('\\(((?:[0-9]+,)+[0-9]+)\\)', eq_atom_text.replace(';', ','))
            eq_atoms = tuple((tuple((int(idx) for idx in t.split(','))) for t in eq_tokens))
        return (heavy_atom_labels, eq_atoms, inchi)

    @staticmethod
    def _group_centroid(mol, ilabels, group_atoms):
        """
        Calculate the centroids of a group atoms indexed by the labels of inchi.

        Args:
            mol: The molecule. OpenBabel OBMol object
            ilabel: inchi label map

        Returns:
            Centroid. Tuple (x, y, z)
        """
        c1x, c1y, c1z = (0.0, 0.0, 0.0)
        for idx in group_atoms:
            orig_idx = ilabels[idx - 1]
            oa1 = mol.GetAtom(orig_idx)
            c1x += float(oa1.x())
            c1y += float(oa1.y())
            c1z += float(oa1.z())
        n_atoms = len(group_atoms)
        c1x /= n_atoms
        c1y /= n_atoms
        c1z /= n_atoms
        return (c1x, c1y, c1z)

    def _virtual_molecule(self, mol, ilabels, eq_atoms):
        """
        Create a virtual molecule by unique atoms, the centroids of the
        equivalent atoms.

        Args:
            mol: The molecule. OpenBabel OBMol object
            ilabels: inchi label map
            eq_atoms: equivalent atom labels
            farthest_group_idx: The equivalent atom group index in which
                there is the farthest atom to the centroid

        Returns:
            The virtual molecule
        """
        vmol = openbabel.OBMol()
        non_unique_atoms = {a for g in eq_atoms for a in g}
        all_atoms = set(range(1, len(ilabels) + 1))
        unique_atom_labels = sorted(all_atoms - non_unique_atoms)
        for idx in unique_atom_labels:
            orig_idx = ilabels[idx - 1]
            oa1 = mol.GetAtom(orig_idx)
            a1 = vmol.NewAtom()
            a1.SetAtomicNum(oa1.GetAtomicNum())
            a1.SetVector(oa1.GetVector())
        if vmol.NumAtoms() < 3:
            for symm in eq_atoms:
                c1x, c1y, c1z = self._group_centroid(mol, ilabels, symm)
                min_distance = float('inf')
                for idx in range(1, vmol.NumAtoms() + 1):
                    va = vmol.GetAtom(idx)
                    distance = math.sqrt((c1x - va.x()) ** 2 + (c1y - va.y()) ** 2 + (c1z - va.z()) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                if min_distance > 0.2:
                    a1 = vmol.NewAtom()
                    a1.SetAtomicNum(9)
                    a1.SetVector(c1x, c1y, c1z)
        return vmol

    @staticmethod
    def _align_heavy_atoms(mol1, mol2, vmol1, vmol2, ilabel1, ilabel2, eq_atoms):
        """
        Align the label of topologically identical atoms of second molecule
        towards first molecule.

        Args:
            mol1: First molecule. OpenBabel OBMol object
            mol2: Second molecule. OpenBabel OBMol object
            vmol1: First virtual molecule constructed by centroids. OpenBabel
                OBMol object
            vmol2: First virtual molecule constructed by centroids. OpenBabel
                OBMol object
            ilabel1: inchi label map of the first molecule
            ilabel2: inchi label map of the second molecule
            eq_atoms: equivalent atom labels

        Returns:
            corrected inchi labels of heavy atoms of the second molecule
        """
        n_virtual = vmol1.NumAtoms()
        n_heavy = len(ilabel1)
        for idx in ilabel2:
            a1 = vmol1.NewAtom()
            a1.SetAtomicNum(1)
            a1.SetVector(0.0, 0.0, 0.0)
            oa2 = mol2.GetAtom(idx)
            a2 = vmol2.NewAtom()
            a2.SetAtomicNum(1)
            a2.SetVector(oa2.GetVector())
        aligner = openbabel.OBAlign(False, False)
        aligner.SetRefMol(vmol1)
        aligner.SetTargetMol(vmol2)
        aligner.Align()
        aligner.UpdateCoords(vmol2)
        canon_mol1 = openbabel.OBMol()
        for idx in ilabel1:
            oa1 = mol1.GetAtom(idx)
            a1 = canon_mol1.NewAtom()
            a1.SetAtomicNum(oa1.GetAtomicNum())
            a1.SetVector(oa1.GetVector())
        aligned_mol2 = openbabel.OBMol()
        for idx in range(n_virtual + 1, n_virtual + n_heavy + 1):
            oa2 = vmol2.GetAtom(idx)
            a2 = aligned_mol2.NewAtom()
            a2.SetAtomicNum(oa2.GetAtomicNum())
            a2.SetVector(oa2.GetVector())
        canon_label2 = list(range(1, n_heavy + 1))
        for symm in eq_atoms:
            for idx in symm:
                canon_label2[idx - 1] = -1
        for symm in eq_atoms:
            candidates1 = list(symm)
            candidates2 = list(symm)
            for c2 in candidates2:
                distance = 99999.0
                canon_idx = candidates1[0]
                a2 = aligned_mol2.GetAtom(c2)
                for c1 in candidates1:
                    a1 = canon_mol1.GetAtom(c1)
                    dist = a1.GetDistance(a2)
                    if dist < distance:
                        distance = dist
                        canon_idx = c1
                canon_label2[c2 - 1] = canon_idx
                candidates1.remove(canon_idx)
        canon_inchi_orig_map2 = list(zip(canon_label2, list(range(1, n_heavy + 1)), ilabel2))
        canon_inchi_orig_map2.sort(key=lambda m: m[0])
        return tuple((x[2] for x in canon_inchi_orig_map2))

    @staticmethod
    def _align_hydrogen_atoms(mol1, mol2, heavy_indices1, heavy_indices2):
        """
        Align the label of topologically identical atoms of second molecule
        towards first molecule.

        Args:
            mol1: First molecule. OpenBabel OBMol object
            mol2: Second molecule. OpenBabel OBMol object
            heavy_indices1: inchi label map of the first molecule
            heavy_indices2: label map of the second molecule

        Returns:
            corrected label map of all atoms of the second molecule
        """
        num_atoms = mol2.NumAtoms()
        all_atom = set(range(1, num_atoms + 1))
        hydrogen_atoms1 = all_atom - set(heavy_indices1)
        hydrogen_atoms2 = all_atom - set(heavy_indices2)
        label1 = heavy_indices1 + tuple(hydrogen_atoms1)
        label2 = heavy_indices2 + tuple(hydrogen_atoms2)
        cmol1 = openbabel.OBMol()
        for idx in label1:
            oa1 = mol1.GetAtom(idx)
            a1 = cmol1.NewAtom()
            a1.SetAtomicNum(oa1.GetAtomicNum())
            a1.SetVector(oa1.GetVector())
        cmol2 = openbabel.OBMol()
        for idx in label2:
            oa2 = mol2.GetAtom(idx)
            a2 = cmol2.NewAtom()
            a2.SetAtomicNum(oa2.GetAtomicNum())
            a2.SetVector(oa2.GetVector())
        aligner = openbabel.OBAlign(False, False)
        aligner.SetRefMol(cmol1)
        aligner.SetTargetMol(cmol2)
        aligner.Align()
        aligner.UpdateCoords(cmol2)
        hydrogen_label2 = []
        hydrogen_label1 = list(range(len(heavy_indices1) + 1, num_atoms + 1))
        for h2 in range(len(heavy_indices2) + 1, num_atoms + 1):
            distance = 99999.0
            idx = hydrogen_label1[0]
            a2 = cmol2.GetAtom(h2)
            for h1 in hydrogen_label1:
                a1 = cmol1.GetAtom(h1)
                dist = a1.GetDistance(a2)
                if dist < distance:
                    distance = dist
                    idx = h1
            hydrogen_label2.append(idx)
            hydrogen_label1.remove(idx)
        hydrogen_orig_idx2 = label2[len(heavy_indices2):]
        hydrogen_canon_orig_map2 = list(zip(hydrogen_label2, hydrogen_orig_idx2))
        hydrogen_canon_orig_map2.sort(key=lambda m: m[0])
        hydrogen_canon_indices2 = [x[1] for x in hydrogen_canon_orig_map2]
        canon_label1 = label1
        canon_label2 = heavy_indices2 + tuple(hydrogen_canon_indices2)
        return (canon_label1, canon_label2)

    @staticmethod
    def _get_elements(mol, label):
        """
        The elements of the atoms in the specified order.

        Args:
            mol: The molecule. OpenBabel OBMol object.
            label: The atom indices. List of integers.

        Returns:
            Elements. List of integers.
        """
        return [int(mol.GetAtom(idx).GetAtomicNum()) for idx in label]

    def _is_molecule_linear(self, mol):
        """
        Is the molecule a linear one.

        Args:
            mol: The molecule. OpenBabel OBMol object.

        Returns:
            Boolean value.
        """
        if mol.NumAtoms() < 3:
            return True
        a1 = mol.GetAtom(1)
        a2 = mol.GetAtom(2)
        for idx in range(3, mol.NumAtoms() + 1):
            angle = float(mol.GetAtom(idx).GetAngle(a2, a1))
            if angle < 0.0:
                angle = -angle
            if angle > 90.0:
                angle = 180.0 - angle
            if angle > self._angle_tolerance:
                return False
        return True

    def uniform_labels(self, mol1, mol2):
        """
        Args:
            mol1 (Molecule): Molecule 1
            mol2 (Molecule): Molecule 2.

        Returns:
            Labels
        """
        ob_mol1 = BabelMolAdaptor(mol1).openbabel_mol
        ob_mol2 = BabelMolAdaptor(mol2).openbabel_mol
        ilabel1, iequal_atom1, inchi1 = self._inchi_labels(ob_mol1)
        ilabel2, iequal_atom2, inchi2 = self._inchi_labels(ob_mol2)
        if inchi1 != inchi2:
            return (None, None)
        if iequal_atom1 != iequal_atom2:
            raise RuntimeError('Design Error! Equivalent atoms are inconsistent')
        vmol1 = self._virtual_molecule(ob_mol1, ilabel1, iequal_atom1)
        vmol2 = self._virtual_molecule(ob_mol2, ilabel2, iequal_atom2)
        if vmol1.NumAtoms() != vmol2.NumAtoms():
            return (None, None)
        if vmol1.NumAtoms() < 3 or self._is_molecule_linear(vmol1) or self._is_molecule_linear(vmol2):
            c_label1, c_label2 = self._assistant_mapper.uniform_labels(mol1, mol2)
        else:
            heavy_atom_indices2 = self._align_heavy_atoms(ob_mol1, ob_mol2, vmol1, vmol2, ilabel1, ilabel2, iequal_atom1)
            c_label1, c_label2 = self._align_hydrogen_atoms(ob_mol1, ob_mol2, ilabel1, heavy_atom_indices2)
        if c_label1 and c_label2:
            elements1 = self._get_elements(ob_mol1, c_label1)
            elements2 = self._get_elements(ob_mol2, c_label2)
            if elements1 != elements2:
                return (None, None)
        return (c_label1, c_label2)

    def get_molecule_hash(self, mol):
        """Return inchi as molecular hash."""
        ob_mol = BabelMolAdaptor(mol).openbabel_mol
        return self._inchi_labels(ob_mol)[2]