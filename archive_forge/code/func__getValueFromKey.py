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
def _getValueFromKey(d, key):
    value = getattr(d, key)
    return value() if callable(value) else value