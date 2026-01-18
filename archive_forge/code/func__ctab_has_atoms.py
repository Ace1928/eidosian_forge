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
def _ctab_has_atoms(ctab_lines):
    """ look at atom count position (line 4, characters 0:3)
    Return True if the count is > 0, False if 0.
    Throw BadMoleculeException if there are no characters
    at the required position or if they cannot be converted
    to a positive integer
    """
    try:
        a_count = int(ctab_lines[3][0:3])
        if a_count < 0:
            raise BadMoleculeException('Atom count negative')
        return a_count > 0
    except IndexError:
        raise BadMoleculeException('Invalid molfile format')
    except ValueError:
        raise BadMoleculeException(f'Expected integer')