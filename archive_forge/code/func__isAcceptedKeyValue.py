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
def _isAcceptedKeyValue(key, value):
    return not key.startswith('__') and (type(value) in (bool, int, float, str, tuple, list) or (callable(value) and key.startswith('get')))