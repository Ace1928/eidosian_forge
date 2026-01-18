import pickle
from rdkit import Chem, DataStructs
@staticmethod
def GetFingerprint(mol, query=True):
    if query:
        flags = LayeredOptions.searchLayerFlags
    else:
        flags = LayeredOptions.loadLayerFlags
    return Chem.LayeredFingerprint(mol, layerFlags=flags, minPath=LayeredOptions.minPath, maxPath=LayeredOptions.maxPath, fpSize=LayeredOptions.fpSize)