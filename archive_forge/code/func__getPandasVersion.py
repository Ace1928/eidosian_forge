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
def _getPandasVersion():
    """ Get the pandas version as a tuple """
    import re
    try:
        v = pd.__version__
    except AttributeError:
        v = pd.version.version
    v = re.split('[^0-9,.]', v)[0].split('.')
    return tuple((int(vi) for vi in v))