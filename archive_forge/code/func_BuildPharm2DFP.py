import pickle
from rdkit import Chem, DataStructs
def BuildPharm2DFP(mol):
    global sigFactory
    from rdkit.Chem.Pharm2D import Generate
    try:
        fp = Generate.Gen2DFingerprint(mol, sigFactory)
    except IndexError:
        print('FAIL:', Chem.MolToSmiles(mol, True))
        raise
    return fp