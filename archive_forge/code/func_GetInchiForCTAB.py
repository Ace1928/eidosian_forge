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
def GetInchiForCTAB(ctab):
    """
    >>> from rdkit.Chem.MolKey import MolKey
    >>> from rdkit.Avalon import pyAvalonTools
    >>> res = MolKey.GetInchiForCTAB(pyAvalonTools.Generate2DCoords('c1cn[nH]c1C(Cl)Br',True))
    >>> res.inchi
    'InChI=1/C4H4BrClN2/c5-4(6)3-1-2-7-8-3/h1-2,4H,(H,7,8)/t4?/f/h8H'
    >>> res = MolKey.GetInchiForCTAB(pyAvalonTools.Generate2DCoords('c1c[nH]nc1C(Cl)Br',True))
    >>> res.inchi
    'InChI=1/C4H4BrClN2/c5-4(6)3-1-2-7-8-3/h1-2,4H,(H,7,8)/t4?/f/h7H'
    >>>
    """
    inchi = None
    strucheck_err, fixed_mol = CheckCTAB(ctab, False)
    if strucheck_err & BAD_SET:
        return InchiResult(strucheck_err, None, fixed_mol)
    conversion_err = 0
    try:
        r_mol = Chem.MolFromMolBlock(fixed_mol, sanitize=False)
        if r_mol:
            inchi = Chem.MolToInchi(r_mol, '/FixedH /SUU')
            if not inchi:
                conversion_err = INCHI_COMPUTATION_ERROR
        else:
            conversion_err = RDKIT_CONVERSION_ERROR
    except Chem.InchiReadWriteError:
        conversion_err = INCHI_READWRITE_ERROR
    return InchiResult(strucheck_err | conversion_err, inchi, fixed_mol)