import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def _getSmartsSaltsFromFile(filename):
    """
    Extracts SMARTS salts from given file object.
    """
    return _getSmartsSaltsFromStream(open(filename, 'r'))