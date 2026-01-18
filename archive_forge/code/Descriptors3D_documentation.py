from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import _isCallable

    Compute all 3D descriptors of a molecule
    
    Arguments:
    - mol: the molecule to work with
    - confId: conformer ID to work with. If not specified the default (-1) is used
    
    Return:
    
    dict
        A dictionary with decriptor names as keys and the descriptor values as values

    raises a ValueError 
        If the molecule does not have conformers
    