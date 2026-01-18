import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def _isDrawOptionEqual(v1, v2):
    if type(v1) != type(v2):
        return False
    if type(v1) in (tuple, list):
        if len(v1) != len(v2):
            return False
        return all((_isDrawOptionEqual(item1, v2[i]) for i, item1 in enumerate(v1)))
    if type(v1) == float:
        return abs(v1 - v2) < 1e-05
    return v1 == v2