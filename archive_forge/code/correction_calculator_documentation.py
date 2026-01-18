from __future__ import annotations
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from monty.serialization import loadfn
from ruamel import yaml
from scipy.optimize import curve_fit
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.structure_analyzer import sulfide_type
from pymatgen.core import Composition, Element
Creates the _name_Compatibility.yaml that stores corrections as well as _name_CompatibilityUncertainties.yaml
        for correction uncertainties.

        Args:
            name: str, alternate name for the created .yaml file.
                Default: "MP2020"
            dir: str, directory in which to save the file. Pass None (default) to
                save the file in the current working directory.
        