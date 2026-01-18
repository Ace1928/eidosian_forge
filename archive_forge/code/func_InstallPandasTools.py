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
def InstallPandasTools():
    """ Monkey patch an RDKit method of Chem.Mol and pandas """
    try:
        PandasPatcher.patchPandas()
    except NameError:
        pass
    if 'Chem.Mol.__ge__' not in _originalSettings:
        _originalSettings['Chem.Mol.__ge__'] = rdchem.Mol.__ge__
        rdchem.Mol.__ge__ = _molge