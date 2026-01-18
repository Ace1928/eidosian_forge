from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
@staticmethod
def _compute_sign(ptolemy_index, perm):
    """
        This is reimplementing _compute_sign
        from addl_code/ptolemy_equations.c
        """
    effective_perm = []
    for v in range(4):
        if ptolemy_index[v] % 2:
            effective_perm.append(perm[v])
    if len(effective_perm) < 2:
        return +1
    if len(effective_perm) == 2:
        if effective_perm[0] < effective_perm[1]:
            return +1
        return -1
    if len(effective_perm) == 3:
        for i in range(3):
            if effective_perm[i] < effective_perm[(i + 1) % 3] < effective_perm[(i + 2) % 3]:
                return +1
        return -1
    raise Exception('Should never reach here')