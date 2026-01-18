import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
Write ensemble data file