from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
@staticmethod
def _get_icohps(icohpcollection, isite, lowerlimit, upperlimit, only_bonds_to):
    """Return icohp dict for certain site.

        Args:
            icohpcollection: Icohpcollection object
            isite (int): number of a site
            lowerlimit (float): lower limit that tells you which ICOHPs are considered
            upperlimit (float): upper limit that tells you which ICOHPs are considered
            only_bonds_to (list): list of str, e.g. ["O"] that will ensure that only bonds to "O" will be considered

        Returns:
            dict: of IcohpValues. The keys correspond to the values from the initial list_labels.
        """
    return icohpcollection.get_icohp_dict_of_site(site=isite, maxbondlength=6.0, minsummedicohp=lowerlimit, maxsummedicohp=upperlimit, only_bonds_to=only_bonds_to)