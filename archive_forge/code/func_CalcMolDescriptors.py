from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def CalcMolDescriptors(mol, missingVal=None, silent=True):
    """ calculate the full set of descriptors for a molecule
    
    Parameters
    ----------
    mol : RDKit molecule
    missingVal : float, optional
                 This will be used if a particular descriptor cannot be calculated
    silent : bool, optional
             if True then exception messages from descriptors will be displayed

    Returns
    -------
    dict 
         A dictionary with decriptor names as keys and the descriptor values as values
    """
    res = {}
    for nm, fn in _descList:
        try:
            val = fn(mol)
        except:
            if not silent:
                import traceback
                traceback.print_exc()
            val = missingVal
        res[nm] = val
    return res