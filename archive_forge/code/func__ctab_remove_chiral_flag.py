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
def _ctab_remove_chiral_flag(ctab_lines):
    """ read the chiral flag (line 4, characters 12:15)
    and set it to 0. Return True if it was 1, False if 0.
    Throw BadMoleculeException if there are no characters
    at the required position or if they where not 0 or 1
    """
    str_a_count = ctab_lines[3][12:15]
    try:
        a_count = int(str_a_count)
        if a_count == 0:
            rval = False
        elif a_count == 1:
            rval = True
            orig_line = ctab_lines[3]
            ctab_lines[3] = orig_line[:CHIRAL_POS] + '  0' + orig_line[CHIRAL_POS + 3:]
        else:
            raise BadMoleculeException('Expected chiral flag 0 or 1')
    except IndexError:
        raise BadMoleculeException('Invalid molfile format')
    except ValueError:
        raise BadMoleculeException(f'Expected integer, got {str_a_count}')
    return rval