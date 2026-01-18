import pickle
from rdkit import Chem, DataStructs
@staticmethod
def GetWords(mol, query=True):
    txt = LayeredOptions.GetFingerprint(mol, query=query).ToBitString()
    return [int(txt[x:x + 32], 2) for x in range(0, len(txt), 32)]