import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def GetLeaves(self):
    """ returns a dictionary, keyed by SMILES, of leaf (terminal) nodes """
    res = {}
    for smi, child in self.children.items():
        if not len(child.children):
            res[smi] = child
        else:
            child._gacRecurse(res, terminalOnly=True)
    return res