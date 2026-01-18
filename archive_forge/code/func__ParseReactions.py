import collections
from rdkit import Chem
from rdkit.Chem import AllChem
def _ParseReactions():
    for row in GetHeterocycleReactionSmarts():
        smarts = row.SMARTS
        if not smarts:
            continue
        for product in row.CONVERT_TO.split(','):
            reaction = smarts + '>>' + product
            yield AllChem.ReactionFromSmarts(reaction)