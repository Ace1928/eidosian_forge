from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
def _sanitize_symbol(symbol):
    if symbol == 'vacancy':
        symbol = DummySpecies('X_vacancy', oxidation_state=None)
    elif symbol == 'X':
        symbol = DummySpecies('X', oxidation_state=None)
    return symbol