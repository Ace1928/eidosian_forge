from __future__ import annotations
import fractions
import itertools
import logging
import math
import re
import subprocess
from glob import glob
from shutil import which
from threading import Timer
import numpy as np
from monty.dev import requires
from monty.fractions import lcm
from monty.tempfile import ScratchDir
from pymatgen.core import DummySpecies, PeriodicSite, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        Enumlib doesn"t work when the number of species get too large. To
        simplify matters, we generate the input file only with disordered sites
        and exclude the ordered sites from the enumeration. The fact that
        different disordered sites with the exact same species may belong to
        different equivalent sites is dealt with by having determined the
        spacegroup earlier and labelling the species differently.
        