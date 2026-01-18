from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict
Calculates the neighbour score for a single atom in a molecule
        The second power allows to account for branching in the molecular structure