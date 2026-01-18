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
def RenderImagesInAllDataFrames(images=True):
    """Changes the default dataframe rendering to not escape HTML characters, thus allowing
    rendered images in all dataframes.
    IMPORTANT: THIS IS A GLOBAL CHANGE THAT WILL AFFECT TO COMPLETE PYTHON SESSION. If you want
    to change the rendering only for a single dataframe use the "ChangeMoleculeRendering" method
    instead.
    """
    try:
        PandasPatcher.renderImagesInAllDataFrames(images)
    except NameError:
        log.warning('Failed to patch pandas - unable to change molecule rendering')