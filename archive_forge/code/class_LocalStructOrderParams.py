from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
class LocalStructOrderParams:
    """
    This class permits the calculation of various types of local
    structure order parameters.
    """
    __supported_types = ('cn', 'sgl_bd', 'bent', 'tri_plan', 'tri_plan_max', 'reg_tri', 'sq_plan', 'sq_plan_max', 'pent_plan', 'pent_plan_max', 'sq', 'tet', 'tet_max', 'tri_pyr', 'sq_pyr', 'sq_pyr_legacy', 'tri_bipyr', 'sq_bipyr', 'oct', 'oct_legacy', 'pent_pyr', 'hex_pyr', 'pent_bipyr', 'hex_bipyr', 'T', 'cuboct', 'cuboct_max', 'see_saw_rect', 'bcc', 'q2', 'q4', 'q6', 'oct_max', 'hex_plan_max', 'sq_face_cap_trig_pris')

    def __init__(self, types, parameters=None, cutoff=-10.0) -> None:
        """
        Args:
            types ([string]): list of strings representing the types of
                order parameters to be calculated. Note that multiple
                mentions of the same type may occur. Currently available
                types recognize following environments:
                  "cn": simple coordination number---normalized
                        if desired;
                  "sgl_bd": single bonds;
                  "bent": bent (angular) coordinations
                          (Zimmermann & Jain, in progress, 2017);
                  "T": T-shape coordinations;
                  "see_saw_rect": see saw-like coordinations;
                  "tet": tetrahedra
                         (Zimmermann et al., submitted, 2017);
                  "oct": octahedra
                         (Zimmermann et al., submitted, 2017);
                  "bcc": body-centered cubic environments (Peters,
                         J. Chem. Phys., 131, 244103, 2009);
                  "tri_plan": trigonal planar environments;
                  "sq_plan": square planar environments;
                  "pent_plan": pentagonal planar environments;
                  "tri_pyr": trigonal pyramids (coordinated atom is in
                             the center of the basal plane);
                  "sq_pyr": square pyramids;
                  "pent_pyr": pentagonal pyramids;
                  "hex_pyr": hexagonal pyramids;
                  "tri_bipyr": trigonal bipyramids;
                  "sq_bipyr": square bipyramids;
                  "pent_bipyr": pentagonal bipyramids;
                  "hex_bipyr": hexagonal bipyramids;
                  "cuboct": cuboctahedra;
                  "q2": motif-unspecific bond orientational order
                        parameter (BOOP) of weight l=2 (Steinhardt
                        et al., Phys. Rev. B, 28, 784-805, 1983);
                  "q4": BOOP of weight l=4;
                  "q6": BOOP of weight l=6.
                  "reg_tri": regular triangle with varying height
                             to basal plane;
                  "sq": square coordination (cf., "reg_tri");
                  "oct_legacy": original Peters-style OP recognizing
                                octahedral coordination environments
                                (Zimmermann et al., J. Am. Chem. Soc.,
                                137, 13352-13361, 2015) that can, however,
                                produce small negative values sometimes.
                  "sq_pyr_legacy": square pyramids (legacy);
            parameters ([dict]): list of dictionaries
                that store float-type parameters associated with the
                definitions of the different order parameters
                (length of list = number of OPs). If an entry
                is None, default values are used that are read from
                the op_params.yaml file. With few exceptions, 9 different
                parameters are used across all OPs:
                  "norm": normalizing constant (used in "cn"
                      (default value: 1)).
                  "TA": target angle (TA) in fraction of 180 degrees
                      ("bent" (1), "tet" (0.6081734479693927),
                      "tri_plan" (0.66666666667), "pent_plan" (0.6),
                      "sq_pyr_legacy" (0.5)).
                  "IGW_TA": inverse Gaussian width (IGW) for penalizing
                      angles away from the target angle in inverse
                      fractions of 180 degrees to ("bent" and "tet" (15),
                      "tri_plan" (13.5), "pent_plan" (18),
                      "sq_pyr_legacy" (30)).
                  "IGW_EP": IGW for penalizing angles away from the
                      equatorial plane (EP) at 90 degrees ("T", "see_saw_rect",
                      "oct", "sq_plan", "tri_pyr", "sq_pyr", "pent_pyr",
                      "hex_pyr", "tri_bipyr", "sq_bipyr", "pent_bipyr",
                      "hex_bipyr", and "oct_legacy" (18)).
                  "fac_AA": factor applied to azimuth angle (AA) in cosine
                      term ("T", "tri_plan", and "sq_plan" (1), "tet",
                      "tri_pyr", and "tri_bipyr" (1.5), "oct", "sq_pyr",
                      "sq_bipyr", and "oct_legacy" (2), "pent_pyr"
                      and "pent_bipyr" (2.5), "hex_pyr" and
                      "hex_bipyr" (3)).
                  "exp_cos_AA": exponent applied to cosine term of AA
                      ("T", "tet", "oct", "tri_plan", "sq_plan",
                      "tri_pyr", "sq_pyr", "pent_pyr", "hex_pyr",
                      "tri_bipyr", "sq_bipyr", "pent_bipyr", "hex_bipyr",
                      and "oct_legacy" (2)).
                  "min_SPP": smallest angle (in radians) to consider
                      a neighbor to be
                      at South pole position ("see_saw_rect", "oct", "bcc",
                      "sq_plan", "tri_bipyr", "sq_bipyr", "pent_bipyr",
                      "hex_bipyr", "cuboct", and "oct_legacy"
                      (2.792526803190927)).
                  "IGW_SPP": IGW for penalizing angles away from South
                      pole position ("see_saw_rect", "oct", "bcc", "sq_plan",
                      "tri_bipyr", "sq_bipyr", "pent_bipyr", "hex_bipyr",
                      "cuboct", and "oct_legacy" (15)).
                  "w_SPP": weight for South pole position relative to
                      equatorial positions ("see_saw_rect" and "sq_plan" (1),
                      "cuboct" (1.8), "tri_bipyr" (2), "oct",
                      "sq_bipyr", and "oct_legacy" (3), "pent_bipyr" (4),
                      "hex_bipyr" (5), "bcc" (6)).
            cutoff (float): Cutoff radius to determine which nearest
                neighbors are supposed to contribute to the order
                parameters. If the value is negative the neighboring
                sites found by distance and cutoff radius are further
                pruned using the get_nn method from the
                VoronoiNN class.
        """
        for typ in types:
            if typ not in LocalStructOrderParams.__supported_types:
                raise ValueError(f'Unknown order parameter type ({typ})!')
        self._types = tuple(types)
        self._comp_azi = False
        self._params = []
        for idx, typ in enumerate(self._types):
            dct = deepcopy(default_op_params[typ]) if default_op_params[typ] is not None else None
            if parameters is None or parameters[idx] is None:
                self._params.append(dct)
            else:
                self._params.append(deepcopy(parameters[idx]))
        self._computerijs = self._computerjks = self._geomops = False
        self._geomops2 = self._boops = False
        self._max_trig_order = -1
        if 'sgl_bd' in self._types:
            self._computerijs = True
        if not set(self._types).isdisjoint(['tet', 'oct', 'bcc', 'sq_pyr', 'sq_pyr_legacy', 'tri_bipyr', 'sq_bipyr', 'oct_legacy', 'tri_plan', 'sq_plan', 'pent_plan', 'tri_pyr', 'pent_pyr', 'hex_pyr', 'pent_bipyr', 'hex_bipyr', 'T', 'cuboct', 'oct_max', 'tet_max', 'tri_plan_max', 'sq_plan_max', 'pent_plan_max', 'cuboct_max', 'bent', 'see_saw_rect', 'hex_plan_max', 'sq_face_cap_trig_pris']):
            self._computerijs = self._geomops = True
        if 'sq_face_cap_trig_pris' in self._types:
            self._comp_azi = True
        if not set(self._types).isdisjoint(['reg_tri', 'sq']):
            self._computerijs = self._computerjks = self._geomops2 = True
        if not set(self._types).isdisjoint(['q2', 'q4', 'q6']):
            self._computerijs = self._boops = True
        if 'q2' in self._types:
            self._max_trig_order = 2
        if 'q4' in self._types:
            self._max_trig_order = 4
        if 'q6' in self._types:
            self._max_trig_order = 6
        if cutoff < 0.0:
            self._cutoff = -cutoff
            self._voroneigh = True
        elif cutoff > 0.0:
            self._cutoff = cutoff
            self._voroneigh = False
        else:
            raise ValueError('Cutoff radius is zero!')
        self._last_nneigh = -1
        self._pow_sin_t: dict[int, list[float]] = {}
        self._pow_cos_t: dict[int, list[float]] = {}
        self._sin_n_p: dict[int, list[float]] = {}
        self._cos_n_p: dict[int, list[float]] = {}

    @property
    def num_ops(self):
        """
        Returns:
            int: the number of different order parameters that are targeted to be calculated.
        """
        return len(self._types)

    @property
    def last_nneigh(self):
        """
        Returns:
            int: the number of neighbors encountered during the most
                recent order parameter calculation. A value of -1 indicates
                that no such calculation has yet been performed for this instance.
        """
        return len(self._last_nneigh)

    def compute_trigonometric_terms(self, thetas, phis):
        """
        Computes trigonometric terms that are required to
        calculate bond orientational order parameters using
        internal variables.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.
                The list of
                azimuth angles of all neighbors in radians. The list of
                azimuth angles is expected to have the same size as the
                list of polar angles; otherwise, a ValueError is raised.
                Also, the two lists of angles have to be coherent in
                order. That is, it is expected that the order in the list
                of azimuth angles corresponds to a distinct sequence of
                neighbors. And, this sequence has to equal the sequence
                of neighbors in the list of polar angles.
        """
        if len(thetas) != len(phis):
            raise ValueError('List of polar and azimuthal angles have to be equal!')
        self._pow_sin_t.clear()
        self._pow_cos_t.clear()
        self._sin_n_p.clear()
        self._cos_n_p.clear()
        self._pow_sin_t[1] = [sin(float(t)) for t in thetas]
        self._pow_cos_t[1] = [cos(float(t)) for t in thetas]
        self._sin_n_p[1] = [sin(float(p)) for p in phis]
        self._cos_n_p[1] = [cos(float(p)) for p in phis]
        for idx in range(2, self._max_trig_order + 1):
            self._pow_sin_t[idx] = [e[0] * e[1] for e in zip(self._pow_sin_t[idx - 1], self._pow_sin_t[1])]
            self._pow_cos_t[idx] = [e[0] * e[1] for e in zip(self._pow_cos_t[idx - 1], self._pow_cos_t[1])]
            self._sin_n_p[idx] = [sin(float(idx) * float(p)) for p in phis]
            self._cos_n_p[idx] = [cos(float(idx) * float(p)) for p in phis]

    def get_q2(self, thetas=None, phis=None):
        """
        Calculates the value of the bond orientational order parameter of
        weight l=2. If the function is called with non-empty lists of
        polar and azimuthal angles the corresponding trigonometric terms
        are computed afresh. Otherwise, it is expected that the
        compute_trigonometric_terms function has been just called.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.

        Returns:
            float: bond orientational order parameter of weight l=2
                corresponding to the input angles thetas and phis.
        """
        if thetas is not None and phis is not None:
            self.compute_trigonometric_terms(thetas, phis)
        n_nn = len(self._pow_sin_t[1])
        n_nn_range = range(n_nn)
        sqrt_15_2pi = sqrt(15 / (2 * pi))
        sqrt_5_pi = sqrt(5 / pi)
        pre_y_2_2 = [0.25 * sqrt_15_2pi * val for val in self._pow_sin_t[2]]
        pre_y_2_1 = [0.5 * sqrt_15_2pi * val[0] * val[1] for val in zip(self._pow_sin_t[1], self._pow_cos_t[1])]
        acc = 0.0
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_2_2[idx] * self._cos_n_p[2][idx]
            imag -= pre_y_2_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_2_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_2_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += 0.25 * sqrt_5_pi * (3 * self._pow_cos_t[2][idx] - 1.0)
        acc += real * real
        real = imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_2_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_2_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_2_2[idx] * self._cos_n_p[2][idx]
            imag += pre_y_2_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        return sqrt(4 * pi * acc / (5 * float(n_nn * n_nn)))

    def get_q4(self, thetas=None, phis=None):
        """
        Calculates the value of the bond orientational order parameter of
        weight l=4. If the function is called with non-empty lists of
        polar and azimuthal angles the corresponding trigonometric terms
        are computed afresh. Otherwise, it is expected that the
        compute_trigonometric_terms function has been just called.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.

        Returns:
            float: bond orientational order parameter of weight l=4
                corresponding to the input angles thetas and phis.
        """
        if thetas is not None and phis is not None:
            self.compute_trigonometric_terms(thetas, phis)
        n_nn = len(self._pow_sin_t[1])
        n_nn_range = range(n_nn)
        i16_3 = 3 / 16.0
        i8_3 = 3 / 8.0
        sqrt_35_pi = sqrt(35 / pi)
        sqrt_35_2pi = sqrt(35 / (2 * pi))
        sqrt_5_pi = sqrt(5 / pi)
        sqrt_5_2pi = sqrt(5 / (2 * pi))
        sqrt_1_pi = sqrt(1 / pi)
        pre_y_4_4 = [i16_3 * sqrt_35_2pi * val for val in self._pow_sin_t[4]]
        pre_y_4_3 = [i8_3 * sqrt_35_pi * val[0] * val[1] for val in zip(self._pow_sin_t[3], self._pow_cos_t[1])]
        pre_y_4_2 = [i8_3 * sqrt_5_2pi * val[0] * (7 * val[1] - 1.0) for val in zip(self._pow_sin_t[2], self._pow_cos_t[2])]
        pre_y_4_1 = [i8_3 * sqrt_5_pi * val[0] * (7 * val[1] - 3 * val[2]) for val in zip(self._pow_sin_t[1], self._pow_cos_t[3], self._pow_cos_t[1])]
        acc = 0.0
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_4[idx] * self._cos_n_p[4][idx]
            imag -= pre_y_4_4[idx] * self._sin_n_p[4][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_3[idx] * self._cos_n_p[3][idx]
            imag -= pre_y_4_3[idx] * self._sin_n_p[3][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_2[idx] * self._cos_n_p[2][idx]
            imag -= pre_y_4_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_4_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += i16_3 * sqrt_1_pi * (35 * self._pow_cos_t[4][idx] - 30 * self._pow_cos_t[2][idx] + 3.0)
        acc += real * real
        real = imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_4_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_4_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_2[idx] * self._cos_n_p[2][idx]
            imag += pre_y_4_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_4_3[idx] * self._cos_n_p[3][idx]
            imag -= pre_y_4_3[idx] * self._sin_n_p[3][idx]
        acc += real * real + imag * imag
        real = imag = 0.0
        for idx in n_nn_range:
            real += pre_y_4_4[idx] * self._cos_n_p[4][idx]
            imag += pre_y_4_4[idx] * self._sin_n_p[4][idx]
        acc += real * real + imag * imag
        return sqrt(4 * pi * acc / (9 * float(n_nn * n_nn)))

    def get_q6(self, thetas=None, phis=None):
        """
        Calculates the value of the bond orientational order parameter of
        weight l=6. If the function is called with non-empty lists of
        polar and azimuthal angles the corresponding trigonometric terms
        are computed afresh. Otherwise, it is expected that the
        compute_trigonometric_terms function has been just called.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.

        Returns:
            float: bond orientational order parameter of weight l=6
                corresponding to the input angles thetas and phis.
        """
        if thetas is not None and phis is not None:
            self.compute_trigonometric_terms(thetas, phis)
        n_nn = len(self._pow_sin_t[1])
        n_nn_range = range(n_nn)
        i64 = 1 / 64.0
        i32 = 1 / 32.0
        i32_3 = 3 / 32.0
        i16 = 1 / 16.0
        sqrt_3003_pi = sqrt(3003 / pi)
        sqrt_1001_pi = sqrt(1001 / pi)
        sqrt_91_2pi = sqrt(91 / (2 * pi))
        sqrt_1365_pi = sqrt(1365 / pi)
        sqrt_273_2pi = sqrt(273 / (2 * pi))
        sqrt_13_pi = sqrt(13 / pi)
        pre_y_6_6 = [i64 * sqrt_3003_pi * val for val in self._pow_sin_t[6]]
        pre_y_6_5 = [i32_3 * sqrt_1001_pi * val[0] * val[1] for val in zip(self._pow_sin_t[5], self._pow_cos_t[1])]
        pre_y_6_4 = [i32_3 * sqrt_91_2pi * val[0] * (11 * val[1] - 1.0) for val in zip(self._pow_sin_t[4], self._pow_cos_t[2])]
        pre_y_6_3 = [i32 * sqrt_1365_pi * val[0] * (11 * val[1] - 3 * val[2]) for val in zip(self._pow_sin_t[3], self._pow_cos_t[3], self._pow_cos_t[1])]
        pre_y_6_2 = [i64 * sqrt_1365_pi * val[0] * (33 * val[1] - 18 * val[2] + 1.0) for val in zip(self._pow_sin_t[2], self._pow_cos_t[4], self._pow_cos_t[2])]
        pre_y_6_1 = [i16 * sqrt_273_2pi * val[0] * (33 * val[1] - 30 * val[2] + 5 * val[3]) for val in zip(self._pow_sin_t[1], self._pow_cos_t[5], self._pow_cos_t[3], self._pow_cos_t[1])]
        acc = 0.0
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_6[idx] * self._cos_n_p[6][idx]
            imag -= pre_y_6_6[idx] * self._sin_n_p[6][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_5[idx] * self._cos_n_p[5][idx]
            imag -= pre_y_6_5[idx] * self._sin_n_p[5][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_4[idx] * self._cos_n_p[4][idx]
            imag -= pre_y_6_4[idx] * self._sin_n_p[4][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_3[idx] * self._cos_n_p[3][idx]
            imag -= pre_y_6_3[idx] * self._sin_n_p[3][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_2[idx] * self._cos_n_p[2][idx]
            imag -= pre_y_6_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_6_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += i32 * sqrt_13_pi * (231 * self._pow_cos_t[6][idx] - 315 * self._pow_cos_t[4][idx] + 105 * self._pow_cos_t[2][idx] - 5.0)
        acc += real * real
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_6_1[idx] * self._cos_n_p[1][idx]
            imag -= pre_y_6_1[idx] * self._sin_n_p[1][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_2[idx] * self._cos_n_p[2][idx]
            imag += pre_y_6_2[idx] * self._sin_n_p[2][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_6_3[idx] * self._cos_n_p[3][idx]
            imag -= pre_y_6_3[idx] * self._sin_n_p[3][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_4[idx] * self._cos_n_p[4][idx]
            imag += pre_y_6_4[idx] * self._sin_n_p[4][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real -= pre_y_6_5[idx] * self._cos_n_p[5][idx]
            imag -= pre_y_6_5[idx] * self._sin_n_p[5][idx]
        acc += real * real + imag * imag
        real = 0.0
        imag = 0.0
        for idx in n_nn_range:
            real += pre_y_6_6[idx] * self._cos_n_p[6][idx]
            imag += pre_y_6_6[idx] * self._sin_n_p[6][idx]
        acc += real * real + imag * imag
        return sqrt(4 * pi * acc / (13 * float(n_nn * n_nn)))

    def get_type(self, index):
        """
        Return type of order parameter at the index provided and
        represented by a short string.

        Args:
            index (int): index of order parameter for which type is
                to be returned.

        Returns:
            str: OP type.
        """
        if index < 0 or index >= len(self._types):
            raise ValueError('Index for getting order parameter type out-of-bounds!')
        return self._types[index]

    def get_parameters(self, index):
        """
        Returns list of floats that represents
        the parameters associated
        with calculation of the order
        parameter that was defined at the index provided.
        Attention: the parameters do not need to equal those originally
        inputted because of processing out of efficiency reasons.

        Args:
            index (int):
                index of order parameter for which associated parameters
                are to be returned.

        Returns:
            [float]: parameters of a given OP.
        """
        if index < 0 or index >= len(self._types):
            raise ValueError('Index for getting parameters associated with order parameter calculation out-of-bounds!')
        return self._params[index]

    def get_order_parameters(self, structure: Structure, n: int, indices_neighs: list[int] | None=None, tol: float=0.0, target_spec=None):
        """
        Compute all order parameters of site n.

        Args:
            structure (Structure): input structure.
            n (int): index of site in input structure,
                for which OPs are to be
                calculated. Note that we do not use the sites iterator
                here, but directly access sites via struct[index].
            indices_neighs (list[int]): list of indices of those neighbors
                in Structure object
                structure that are to be considered for OP computation.
                This optional argument overwrites the way neighbors are
                to be determined as defined in the constructor (i.e.,
                Voronoi coordination finder via negative cutoff radius
                vs constant cutoff radius if cutoff was positive).
                We do not use information about the underlying
                structure lattice if the neighbor indices are explicitly
                provided. This has two important consequences. First,
                the input Structure object can, in fact, be a
                simple list of Site objects. Second, no nearest images
                of neighbors are determined when providing an index list.
                Note furthermore that this neighbor
                determination type ignores the optional target_spec
                argument.
            tol (float): threshold of weight
                (= solid angle / maximal solid angle)
                to determine if a particular pair is
                considered neighbors; this is relevant only in the case
                when Voronoi polyhedra are used to determine coordination
            target_spec (Species): target species to be considered
                when calculating the order
                parameters of site n; None includes all species of input
                structure.

        Returns:
            [floats]: representing order parameters. Should it not be
            possible to compute a given OP for a conceptual reason, the
            corresponding entry is None instead of a float. For Steinhardt
            et al.'s bond orientational OPs and the other geometric OPs
            ("tet", "oct", "bcc", etc.),
            this can happen if there is a single
            neighbor around site n in the structure because that
            does not permit calculation of angles between multiple
            neighbors.
        """
        if n < 0:
            raise ValueError('Site index smaller zero!')
        if n >= len(structure):
            raise ValueError('Site index beyond maximum!')
        if indices_neighs is not None:
            for index in indices_neighs:
                if index >= len(structure):
                    raise ValueError('Neighbor site index beyond maximum!')
        if tol < 0.0:
            raise ValueError('Negative tolerance for weighted solid angle!')
        left_of_unity = 1 - 1e-12
        very_small = 1e-12
        fac_bcc = 1 / exp(-0.5)
        centsite = structure[n]
        if indices_neighs is not None:
            neighsites = [structure[index] for index in indices_neighs]
        elif self._voroneigh:
            vnn = VoronoiNN(tol=tol, targets=target_spec)
            neighsites = vnn.get_nn(structure, n)
        else:
            neighsitestmp = [idx[0] for idx in structure.get_sites_in_sphere(centsite.coords, self._cutoff)]
            neighsites = []
            if centsite not in neighsitestmp:
                raise ValueError('Could not find center site!')
            neighsitestmp.remove(centsite)
            if target_spec is None:
                neighsites = list(neighsitestmp)
            else:
                neighsites[:] = [site for site in neighsitestmp if site.specie.symbol == target_spec]
        n_neighbors = len(neighsites)
        self._last_nneigh = n_neighbors
        rij: list[np.ndarray] = []
        rjk: list[list[np.ndarray]] = []
        rij_norm: list[list[float]] = []
        rjknorm: list[list[np.ndarray]] = []
        dist: list[float] = []
        distjk_unique: list[float] = []
        distjk: list[list[float]] = []
        centvec = centsite.coords
        if self._computerijs:
            for j, neigh in enumerate(neighsites):
                rij.append(neigh.coords - centvec)
                dist.append(float(np.linalg.norm(rij[j])))
                rij_norm.append(rij[j] / dist[j])
        if self._computerjks:
            for j, neigh in enumerate(neighsites):
                rjk.append([])
                rjknorm.append([])
                distjk.append([])
                kk = 0
                for k, neigh_2 in enumerate(neighsites):
                    if j != k:
                        rjk[j].append(neigh_2.coords - neigh.coords)
                        distjk[j].append(float(np.linalg.norm(rjk[j][kk])))
                        if k > j:
                            distjk_unique.append(distjk[j][kk])
                        rjknorm[j].append(rjk[j][kk] / distjk[j][kk])
                        kk = kk + 1
        ops = [0.0 for t in self._types]
        for idx, typ in enumerate(self._types):
            if typ == 'cn':
                ops[idx] = n_neighbors / self._params[idx]['norm']
            elif typ == 'sgl_bd':
                dist_sorted = sorted(dist)
                if len(dist_sorted) == 1:
                    ops[idx] = 1
                elif len(dist_sorted) > 1:
                    ops[idx] = 1 - dist_sorted[0] / dist_sorted[1]
        if self._boops:
            thetas = []
            phis = []
            for vec in rij_norm:
                thetas.append(acos(max(-1.0, min(vec[2], 1.0))))
                tmpphi = 0.0
                if -left_of_unity < vec[2] < left_of_unity:
                    tmpphi = acos(max(-1.0, min(vec[0] / sqrt(vec[0] * vec[0] + vec[1] * vec[1]), 1.0)))
                    if vec[1] < 0.0:
                        tmpphi = -tmpphi
                phis.append(tmpphi)
            for idx, typ in enumerate(self._types):
                if typ == 'q2':
                    ops[idx] = self.get_q2(thetas, phis) if len(thetas) > 0 else None
                elif typ == 'q4':
                    ops[idx] = self.get_q4(thetas, phis) if len(thetas) > 0 else None
                elif typ == 'q6':
                    ops[idx] = self.get_q6(thetas, phis) if len(thetas) > 0 else None
        if self._geomops:
            gaussthetak: list[float] = [0 for t in self._types]
            qsp_theta = [[[] for j in range(n_neighbors)] for t in self._types]
            norms = [[[] for j in range(n_neighbors)] for t in self._types]
            ipi = 1 / pi
            piover2 = pi / 2.0
            onethird = 1 / 3
            twothird = 2 / 3.0
            for j in range(n_neighbors):
                zaxis = rij_norm[j]
                kc = 0
                for k in range(n_neighbors):
                    if j != k:
                        for idx in range(len(self._types)):
                            qsp_theta[idx][j].append(0.0)
                            norms[idx][j].append(0)
                        tmp = max(-1.0, min(np.inner(zaxis, rij_norm[k]), 1.0))
                        thetak = acos(tmp)
                        xaxis = gramschmidt(rij_norm[k], zaxis)
                        if np.linalg.norm(xaxis) < very_small:
                            flag_xaxis = True
                        else:
                            xaxis = xaxis / np.linalg.norm(xaxis)
                            flag_xaxis = False
                        if self._comp_azi:
                            flag_yaxis = True
                            yaxis = np.cross(zaxis, xaxis)
                            if np.linalg.norm(yaxis) > very_small:
                                yaxis = yaxis / np.linalg.norm(yaxis)
                                flag_yaxis = False
                        for idx, typ in enumerate(self._types):
                            if typ in ['bent', 'sq_pyr_legacy']:
                                tmp = self._params[idx]['IGW_TA'] * (thetak * ipi - self._params[idx]['TA'])
                                qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                                norms[idx][j][kc] += 1
                            elif typ in ['tri_plan', 'tri_plan_max', 'tet', 'tet_max']:
                                tmp = self._params[idx]['IGW_TA'] * (thetak * ipi - self._params[idx]['TA'])
                                gaussthetak[idx] = exp(-0.5 * tmp * tmp)
                                if typ in ['tri_plan_max', 'tet_max']:
                                    qsp_theta[idx][j][kc] += gaussthetak[idx]
                                    norms[idx][j][kc] += 1
                            elif typ in ['T', 'tri_pyr', 'sq_pyr', 'pent_pyr', 'hex_pyr']:
                                tmp = self._params[idx]['IGW_EP'] * (thetak * ipi - 0.5)
                                qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                                norms[idx][j][kc] += 1
                            elif typ in ['sq_plan', 'oct', 'oct_legacy', 'cuboct', 'cuboct_max']:
                                if thetak >= self._params[idx]['min_SPP']:
                                    tmp = self._params[idx]['IGW_SPP'] * (thetak * ipi - 1.0)
                                    qsp_theta[idx][j][kc] += self._params[idx]['w_SPP'] * exp(-0.5 * tmp * tmp)
                                    norms[idx][j][kc] += self._params[idx]['w_SPP']
                            elif typ in ['see_saw_rect', 'tri_bipyr', 'sq_bipyr', 'pent_bipyr', 'hex_bipyr', 'oct_max', 'sq_plan_max', 'hex_plan_max']:
                                if thetak < self._params[idx]['min_SPP']:
                                    tmp = self._params[idx]['IGW_EP'] * (thetak * ipi - 0.5) if typ != 'hex_plan_max' else self._params[idx]['IGW_TA'] * (fabs(thetak * ipi - 0.5) - self._params[idx]['TA'])
                                    qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                                    norms[idx][j][kc] += 1
                            elif typ in ['pent_plan', 'pent_plan_max']:
                                tmp = 0.4 if thetak <= self._params[idx]['TA'] * pi else 0.8
                                tmp2 = self._params[idx]['IGW_TA'] * (thetak * ipi - tmp)
                                gaussthetak[idx] = exp(-0.5 * tmp2 * tmp2)
                                if typ == 'pent_plan_max':
                                    qsp_theta[idx][j][kc] += gaussthetak[idx]
                                    norms[idx][j][kc] += 1
                            elif typ == 'bcc' and j < k:
                                if thetak >= self._params[idx]['min_SPP']:
                                    tmp = self._params[idx]['IGW_SPP'] * (thetak * ipi - 1.0)
                                    qsp_theta[idx][j][kc] += self._params[idx]['w_SPP'] * exp(-0.5 * tmp * tmp)
                                    norms[idx][j][kc] += self._params[idx]['w_SPP']
                            elif typ == 'sq_face_cap_trig_pris' and thetak < self._params[idx]['TA3']:
                                tmp = self._params[idx]['IGW_TA1'] * (thetak * ipi - self._params[idx]['TA1'])
                                qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                                norms[idx][j][kc] += 1
                        for m in range(n_neighbors):
                            if m != j and m != k and (not flag_xaxis):
                                tmp = max(-1.0, min(np.inner(zaxis, rij_norm[m]), 1.0))
                                thetam = acos(tmp)
                                x_two_axis_tmp = gramschmidt(rij_norm[m], zaxis)
                                norm = np.linalg.norm(x_two_axis_tmp)
                                if norm < very_small:
                                    flag_xtwoaxis = True
                                else:
                                    xtwoaxis = x_two_axis_tmp / norm
                                    phi = acos(max(-1.0, min(np.inner(xtwoaxis, xaxis), 1.0)))
                                    flag_xtwoaxis = False
                                    if self._comp_azi:
                                        phi2 = atan2(np.dot(xtwoaxis, yaxis), np.dot(xtwoaxis, xaxis))
                                if typ in ['tri_bipyr', 'sq_bipyr', 'pent_bipyr', 'hex_bipyr', 'oct_max', 'sq_plan_max', 'hex_plan_max', 'see_saw_rect'] and thetam >= self._params[idx]['min_SPP']:
                                    tmp = self._params[idx]['IGW_SPP'] * (thetam * ipi - 1.0)
                                    qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                                    norms[idx][j][kc] += 1
                                if not flag_xaxis and (not flag_xtwoaxis):
                                    for idx, typ in enumerate(self._types):
                                        if typ in ['tri_plan', 'tri_plan_max', 'tet', 'tet_max']:
                                            tmp = self._params[idx]['IGW_TA'] * (thetam * ipi - self._params[idx]['TA'])
                                            tmp2 = cos(self._params[idx]['fac_AA'] * phi) ** self._params[idx]['exp_cos_AA']
                                            tmp3 = 1 if typ in ['tri_plan_max', 'tet_max'] else gaussthetak[idx]
                                            qsp_theta[idx][j][kc] += tmp3 * exp(-0.5 * tmp * tmp) * tmp2
                                            norms[idx][j][kc] += 1
                                        elif typ in ['pent_plan', 'pent_plan_max']:
                                            tmp = 0.4 if thetam <= self._params[idx]['TA'] * pi else 0.8
                                            tmp2 = self._params[idx]['IGW_TA'] * (thetam * ipi - tmp)
                                            tmp3 = cos(phi)
                                            tmp4 = 1 if typ == 'pent_plan_max' else gaussthetak[idx]
                                            qsp_theta[idx][j][kc] += tmp4 * exp(-0.5 * tmp2 * tmp2) * tmp3 * tmp3
                                            norms[idx][j][kc] += 1
                                        elif typ in ['T', 'tri_pyr', 'sq_pyr', 'pent_pyr', 'hex_pyr']:
                                            tmp = cos(self._params[idx]['fac_AA'] * phi) ** self._params[idx]['exp_cos_AA']
                                            tmp3 = self._params[idx]['IGW_EP'] * (thetam * ipi - 0.5)
                                            qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp3 * tmp3)
                                            norms[idx][j][kc] += 1
                                        elif typ in ['sq_plan', 'oct', 'oct_legacy']:
                                            if thetak < self._params[idx]['min_SPP'] and thetam < self._params[idx]['min_SPP']:
                                                tmp = cos(self._params[idx]['fac_AA'] * phi) ** self._params[idx]['exp_cos_AA']
                                                tmp2 = self._params[idx]['IGW_EP'] * (thetam * ipi - 0.5)
                                                qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                                if typ == 'oct_legacy':
                                                    qsp_theta[idx][j][kc] -= tmp * self._params[idx][6] * self._params[idx][7]
                                                norms[idx][j][kc] += 1
                                        elif typ in ['tri_bipyr', 'sq_bipyr', 'pent_bipyr', 'hex_bipyr', 'oct_max', 'sq_plan_max', 'hex_plan_max']:
                                            if thetam < self._params[idx]['min_SPP'] and thetak < self._params[idx]['min_SPP']:
                                                tmp = cos(self._params[idx]['fac_AA'] * phi) ** self._params[idx]['exp_cos_AA']
                                                tmp2 = self._params[idx]['IGW_EP'] * (thetam * ipi - 0.5) if typ != 'hex_plan_max' else self._params[idx]['IGW_TA'] * (fabs(thetam * ipi - 0.5) - self._params[idx]['TA'])
                                                qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                                norms[idx][j][kc] += 1
                                        elif typ == 'bcc' and j < k:
                                            if thetak < self._params[idx]['min_SPP']:
                                                fac = 1 if thetak > piover2 else -1
                                                tmp = (thetam - piover2) / asin(1 / 3)
                                                qsp_theta[idx][j][kc] += fac * cos(3 * phi) * fac_bcc * tmp * exp(-0.5 * tmp * tmp)
                                                norms[idx][j][kc] += 1
                                        elif typ == 'see_saw_rect':
                                            if thetam < self._params[idx]['min_SPP'] and thetak < self._params[idx]['min_SPP'] and (phi < 0.75 * pi):
                                                tmp = cos(self._params[idx]['fac_AA'] * phi) ** self._params[idx]['exp_cos_AA']
                                                tmp2 = self._params[idx]['IGW_EP'] * (thetam * ipi - 0.5)
                                                qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                                norms[idx][j][kc] += 1.0
                                        elif typ in ['cuboct', 'cuboct_max']:
                                            if thetam < self._params[idx]['min_SPP'] and self._params[idx][4] < thetak < self._params[idx][2]:
                                                if self._params[idx][4] < thetam < self._params[idx][2]:
                                                    tmp = cos(phi)
                                                    tmp2 = self._params[idx][5] * (thetam * ipi - 0.5)
                                                    qsp_theta[idx][j][kc] += tmp * tmp * exp(-0.5 * tmp2 * tmp2)
                                                    norms[idx][j][kc] += 1.0
                                                elif thetam < self._params[idx][4]:
                                                    tmp = 0.0556 * (cos(phi - 0.5 * pi) - 0.81649658)
                                                    tmp2 = self._params[idx][6] * (thetam * ipi - onethird)
                                                    qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp) * exp(-0.5 * tmp2 * tmp2)
                                                    norms[idx][j][kc] += 1.0
                                                elif thetam > self._params[idx][2]:
                                                    tmp = 0.0556 * (cos(phi - 0.5 * pi) - 0.81649658)
                                                    tmp2 = self._params[idx][6] * (thetam * ipi - twothird)
                                                    qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp) * exp(-0.5 * tmp2 * tmp2)
                                                    norms[idx][j][kc] += 1.0
                                        elif typ == 'sq_face_cap_trig_pris' and (not flag_yaxis) and (thetak < self._params[idx]['TA3']):
                                            if thetam < self._params[idx]['TA3']:
                                                tmp = cos(self._params[idx]['fac_AA1'] * phi2) ** self._params[idx]['exp_cos_AA1']
                                                tmp2 = self._params[idx]['IGW_TA1'] * (thetam * ipi - self._params[idx]['TA1'])
                                            else:
                                                tmp = cos(self._params[idx]['fac_AA2'] * (phi2 + self._params[idx]['shift_AA2'])) ** self._params[idx]['exp_cos_AA2']
                                                tmp2 = self._params[idx]['IGW_TA2'] * (thetam * ipi - self._params[idx]['TA2'])
                                            qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                            norms[idx][j][kc] += 1
                        kc += 1
            for idx, typ in enumerate(self._types):
                if typ in ['tri_plan', 'tet', 'bent', 'sq_plan', 'oct', 'oct_legacy', 'cuboct', 'pent_plan']:
                    ops[idx] = tmp_norm = 0.0
                    for j in range(n_neighbors):
                        ops[idx] += sum(qsp_theta[idx][j])
                        tmp_norm += float(sum(norms[idx][j]))
                    ops[idx] = ops[idx] / tmp_norm if tmp_norm > 1e-12 else None
                elif typ in ['T', 'tri_pyr', 'see_saw_rect', 'sq_pyr', 'tri_bipyr', 'sq_bipyr', 'pent_pyr', 'hex_pyr', 'pent_bipyr', 'hex_bipyr', 'oct_max', 'tri_plan_max', 'tet_max', 'sq_plan_max', 'pent_plan_max', 'cuboct_max', 'hex_plan_max', 'sq_face_cap_trig_pris']:
                    ops[idx] = None
                    if n_neighbors > 1:
                        for j in range(n_neighbors):
                            for k in range(len(qsp_theta[idx][j])):
                                qsp_theta[idx][j][k] = qsp_theta[idx][j][k] / norms[idx][j][k] if norms[idx][j][k] > 1e-12 else 0.0
                            ops[idx] = max(qsp_theta[idx][j]) if j == 0 else max(ops[idx], *qsp_theta[idx][j])
                elif typ == 'bcc':
                    ops[idx] = 0.0
                    for j in range(n_neighbors):
                        ops[idx] += sum(qsp_theta[idx][j])
                    if n_neighbors > 3:
                        ops[idx] = ops[idx] / float(0.5 * float(n_neighbors * (6 + (n_neighbors - 2) * (n_neighbors - 3))))
                    else:
                        ops[idx] = None
                elif typ == 'sq_pyr_legacy':
                    if n_neighbors > 1:
                        dmean = np.mean(dist)
                        acc = 0.0
                        for d in dist:
                            tmp = self._params[idx][2] * (d - dmean)
                            acc = acc + exp(-0.5 * tmp * tmp)
                        for j in range(n_neighbors):
                            ops[idx] = max(qsp_theta[idx][j]) if j == 0 else max(ops[idx], *qsp_theta[idx][j])
                        ops[idx] = acc * ops[idx] / float(n_neighbors)
                    else:
                        ops[idx] = None
        if self._geomops2:
            aij = []
            for ir, r in enumerate(rij_norm, start=1):
                for j in range(ir, len(rij_norm)):
                    aij.append(acos(max(-1.0, min(np.inner(r, rij_norm[j]), 1.0))))
            aijs = sorted(aij)
            neighscent = np.array([0.0, 0.0, 0.0])
            for neigh in neighsites:
                neighscent = neighscent + neigh.coords
            if n_neighbors > 0:
                neighscent = neighscent / float(n_neighbors)
            h = np.linalg.norm(neighscent - centvec)
            b = min(distjk_unique) if len(distjk_unique) > 0 else 0
            dhalf = max(distjk_unique) / 2 if len(distjk_unique) > 0 else 0
            for idx, typ in enumerate(self._types):
                if typ in ('reg_tri', 'sq'):
                    if n_neighbors < 3:
                        ops[idx] = None
                    else:
                        ops[idx] = 1.0
                        if typ == 'reg_tri':
                            a = 2 * asin(b / (2 * sqrt(h * h + (b / (2 * cos(3 * pi / 18))) ** 2)))
                            nmax = 3
                        elif typ == 'sq':
                            a = 2 * asin(b / (2 * sqrt(h * h + dhalf * dhalf)))
                            nmax = 4
                        for j in range(min([n_neighbors, nmax])):
                            ops[idx] = ops[idx] * exp(-0.5 * ((aijs[j] - a) * self._params[idx][0]) ** 2)
        return ops