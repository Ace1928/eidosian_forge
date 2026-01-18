from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import _isCallable
def CalcMolDescriptors3D(mol, confId=None):
    """
    Compute all 3D descriptors of a molecule
    
    Arguments:
    - mol: the molecule to work with
    - confId: conformer ID to work with. If not specified the default (-1) is used
    
    Return:
    
    dict
        A dictionary with decriptor names as keys and the descriptor values as values

    raises a ValueError 
        If the molecule does not have conformers
    """
    if mol.GetNumConformers() == 0:
        raise ValueError('Computing 3D Descriptors requires a structure with at least 1 conformer')
    else:
        vals_3D = {}
        for nm, fn in descList:
            vals_3D[nm] = fn(mol)
        return vals_3D