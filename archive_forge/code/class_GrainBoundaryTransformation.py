from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
class GrainBoundaryTransformation(AbstractTransformation):
    """A transformation that creates a gb from a bulk structure."""

    def __init__(self, rotation_axis, rotation_angle, expand_times=4, vacuum_thickness=0.0, ab_shift=None, normal=False, ratio=True, plane=None, max_search=20, tol_coi=1e-08, rm_ratio=0.7, quick_gen=False):
        """
        Args:
            rotation_axis (list): Rotation axis of GB in the form of a list of integer
                e.g.: [1, 1, 0]
            rotation_angle (float, in unit of degree): rotation angle used to generate GB.
                Make sure the angle is accurate enough. You can use the enum* functions
                in this class to extract the accurate angle.
                e.g.: The rotation angle of sigma 3 twist GB with the rotation axis
                [1, 1, 1] and GB plane (1, 1, 1) can be 60.000000000 degree.
                If you do not know the rotation angle, but know the sigma value, we have
                provide the function get_rotation_angle_from_sigma which is able to return
                all the rotation angles of sigma value you provided.
            expand_times (int): The multiple times used to expand one unit grain to larger grain.
                This is used to tune the grain length of GB to warrant that the two GBs in one
                cell do not interact with each other. Default set to 4.
            vacuum_thickness (float): The thickness of vacuum that you want to insert between
                two grains of the GB. Default to 0.
            ab_shift (list of float, in unit of a, b vectors of Gb): in plane shift of two grains
            normal (logic):
                determine if need to require the c axis of top grain (first transformation matrix)
                perpendicular to the surface or not.
                default to false.
            ratio (list of integers): lattice axial ratio.
                If True, will try to determine automatically from structure.
                For cubic system, ratio is not needed and can be set to None.
                For tetragonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to None.
                For orthorhombic system, ratio = [mu, lam, mv], list of three integers,
                    that is, mu:lam:mv = c2:b2:a2. If irrational for one axis, set it to None.
                e.g. mu:lam:mv = c2,None,a2, means b2 is irrational.
                For rhombohedral system, ratio = [mu, mv], list of two integers,
                that is, mu/mv is the ratio of (1+2*cos(alpha))/cos(alpha).
                If irrational, set it to None.
                For hexagonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.
            plane (list): Grain boundary plane in the form of a list of integers
                e.g.: [1, 2, 3]. If none, we set it as twist GB. The plane will be perpendicular
                to the rotation axis.
            max_search (int): max search for the GB lattice vectors that give the smallest GB
                lattice. If normal is true, also max search the GB c vector that perpendicular
                to the plane. For complex GB, if you want to speed up, you can reduce this value.
                But too small of this value may lead to error.
            tol_coi (float): tolerance to find the coincidence sites. When making approximations to
                the ratio needed to generate the GB, you probably need to increase this tolerance to
                obtain the correct number of coincidence sites. To check the number of coincidence
                sites are correct or not, you can compare the generated Gb object's sigma with enum*
                sigma values (what user expected by input).
            rm_ratio (float): the criteria to remove the atoms which are too close with each other.
                rm_ratio * bond_length of bulk system is the criteria of bond length, below which the atom
                will be removed. Default to 0.7.
            quick_gen (bool): whether to quickly generate a supercell, if set to true, no need to
                find the smallest cell.

        Returns:
            Grain boundary structure (gb (Structure) object).
        """
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.expand_times = expand_times
        self.vacuum_thickness = vacuum_thickness
        self.ab_shift = ab_shift or [0, 0]
        self.normal = normal
        self.ratio = ratio
        self.plane = plane
        self.max_search = max_search
        self.tol_coi = tol_coi
        self.rm_ratio = rm_ratio
        self.quick_gen = quick_gen

    def apply_transformation(self, structure: Structure):
        """Applies the transformation.

        Args:
            structure: Input Structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures
                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Grain boundary Structures.
        """
        gbg = GrainBoundaryGenerator(structure)
        return gbg.gb_from_parameters(self.rotation_axis, self.rotation_angle, expand_times=self.expand_times, vacuum_thickness=self.vacuum_thickness, ab_shift=self.ab_shift, normal=self.normal, ratio=gbg.get_ratio() if self.ratio is True else self.ratio, plane=self.plane, max_search=self.max_search, tol_coi=self.tol_coi, rm_ratio=self.rm_ratio, quick_gen=self.quick_gen)

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False