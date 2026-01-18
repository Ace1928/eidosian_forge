import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def BEEF_Ensemble(*args, **kwargs):
    import warnings
    warnings.warn('Please use BEEFEnsemble instead of BEEF_Ensemble.')
    return BEEFEnsemble(*args, **kwargs)