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
def GetKeyForCTAB(ctab, stereo_info=None, stereo_comment=None, logger=None):
    """
    >>> from rdkit.Chem.MolKey import MolKey
    >>> from rdkit.Avalon import pyAvalonTools
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1ccccc1C(F)Cl', True))
    >>> res.mol_key
    '1|L7676nfGsSIU33wkx//NCg=='
    >>> res.stereo_code
    'R_ONE'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1ccccc1[C@H](F)Cl', True))
    >>> res.mol_key
    '1|Aj38EIxf13RuPDQG2A0UMw=='
    >>> res.stereo_code
    'S_ABS'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1ccccc1[C@@H](F)Cl', True))
    >>> res.mol_key
    '1|9ypfMrhxn1w0ncRooN5HXw=='
    >>> res.stereo_code
    'S_ABS'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1cccc(C(Br)Cl)c1[C@@H](F)Cl', True))
    >>> res.mol_key
    '1|c96jMSlbn7O9GW5d5uB9Mw=='
    >>> res.stereo_code
    'S_PART'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1cccc([C@H](Br)Cl)c1[C@@H](F)Cl', True))
    >>> res.mol_key
    '1|+B+GCEardrJteE8xzYdGLA=='
    >>> res.stereo_code
    'S_ABS'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1cccc(C(Br)Cl)c1C(F)Cl', True))
    >>> res.mol_key
    '1|5H9R3LvclagMXHp3Clrc/g=='
    >>> res.stereo_code
    'S_UNKN'
    >>> res = MolKey.GetKeyForCTAB(pyAvalonTools.Generate2DCoords('c1cccc(C(Br)Cl)c1C(F)Cl',True), stereo_info='S_REL')
    >>> res.mol_key
    '1|cqKWVsUEY6QNpGCbDaDTYA=='
    >>> res.stereo_code
    'S_REL'
    >>> res.inchi
    'InChI=1/C8H6BrCl2F/c9-7(10)5-3-1-2-4-6(5)8(11)12/h1-4,7-8H/t7?,8?'

    """
    if logger is None:
        logger = logging
    try:
        err, inchi, fixed_mol = GetInchiForCTAB(ctab)
    except BadMoleculeException:
        logger.warn(u'Corrupt molecule substituting no-struct: --->\n{0}\n<----'.format(ctab))
        err = NULL_MOL
        key = _identify(err, '', '', None, None)
        return MolKeyResult(key, err, '', '', None, None)
    stereo_category = None
    extra_structure_desc = stereo_comment
    if stereo_info:
        info_flds = stereo_info.split(' ', 1)
        code_fld = info_flds[0]
        if code_fld in stereo_code_dict:
            stereo_category = code_fld
            if not stereo_comment and len(info_flds) > 1:
                extra_structure_desc = info_flds[1].strip()
        else:
            logger.warn(f'stereo code {code_fld} not recognized. Using default value for ctab.')
    if not err & BAD_SET:
        n_stereo, n_undef_stereo, is_meso, dummy = InchiInfo.InchiInfo(inchi).get_sp3_stereo()['main']['non-isotopic']
        if stereo_category is None or stereo_category == 'DEFAULT':
            stereo_category = _get_chiral_identification_string(n_stereo - n_undef_stereo, n_undef_stereo)
    else:
        raise NotImplementedError('currently cannot generate correct keys for molecules with struchk errors')
    key = _identify(err, fixed_mol, inchi, stereo_category, extra_structure_desc)
    return MolKeyResult(key, err, inchi, fixed_mol, stereo_category, extra_structure_desc)