import base64
import hashlib
import logging
import os
import re
import tempfile
import uuid
from collections import namedtuple
from rdkit import Chem, RDConfig
from rdkit.Chem.MolKey import InchiInfo
def _identify(err, ctab, inchi, stereo_category, extra_structure_desc=None):
    """ Compute the molecule key based on the inchi string,
    stereo category as well as extra structure
    information """
    key_string = _get_identification_string(err, ctab, inchi, stereo_category, extra_structure_desc)
    if not key_string:
        return None
    hash_key = base64.b64encode(hashlib.md5(key_string.encode('UTF-8')).digest()).decode()
    return f'{MOL_KEY_VERSION}|{hash_key}'