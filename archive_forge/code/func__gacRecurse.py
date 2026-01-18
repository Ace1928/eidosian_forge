import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def _gacRecurse(self, res, terminalOnly=False):
    for smi, child in self.children.items():
        if not terminalOnly or not len(child.children):
            res[smi] = child
        child._gacRecurse(res, terminalOnly=terminalOnly)