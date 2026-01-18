from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
@property
def _exname(self):
    return Path(self.vib.exname) / f'ex.{self.name}{self.vib.exext}'