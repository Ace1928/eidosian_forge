import pickle
from rdkit import Chem, DataStructs
def BuildAvalonFP(mol, smiles=None):
    from rdkit.Avalon import pyAvalonTools
    if smiles is None:
        return pyAvalonTools.GetAvalonFP(mol)
    return pyAvalonTools.GetAvalonFP(smiles, True)