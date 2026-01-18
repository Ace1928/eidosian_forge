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
def _fix_chemdraw_header(my_string):
    pat = '0V2000'
    sbt = 'V2000'
    return _fix_all(pat, sbt, my_string)