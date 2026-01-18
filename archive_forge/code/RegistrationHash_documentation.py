import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash

    NOTES: if sortAtomAndBondOrder is true then the atom and bond lists will be sorted.
    This assumes that the ordering of those lists is not important
    