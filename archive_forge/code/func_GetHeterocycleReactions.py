import collections
from rdkit import Chem
from rdkit.Chem import AllChem
def GetHeterocycleReactions():
    """
    Return RDKit ChemicalReaction objects of the reaction SMARTS
    returned from GetHeterocyleReactionSmarts.
    """
    global REACTION_CACHE
    if REACTION_CACHE is None:
        REACTION_CACHE = list(_ParseReactions())
    return REACTION_CACHE