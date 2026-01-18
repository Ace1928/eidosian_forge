import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def RemoveSaltsFromFrame(frame, molCol='ROMol'):
    """
    Removes salts from mols in pandas DataFrame's ROMol column
    """
    global _saltRemover
    if _saltRemover is None:
        from rdkit.Chem import SaltRemover
        _saltRemover = SaltRemover.SaltRemover()
    frame[molCol] = frame.apply(lambda x: _saltRemover.StripMol(x[molCol]), axis=1)