import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def _getSmartsSaltsFromStream(stream):
    """
    Yields extracted SMARTS salts from given stream.
    """
    with closing(stream) as lines:
        for line in lines:
            smarts = _smartsFromSmartsLine(line)
            if smarts:
                yield smarts