import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
class EnhancedStereoUpdateMode(enum.Enum):
    ADD_WEIGHTS = enum.auto()
    REMOVE_WEIGHTS = enum.auto()