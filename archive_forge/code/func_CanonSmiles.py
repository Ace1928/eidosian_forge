from rdkit import DataStructs, RDConfig, rdBase
from rdkit.Chem import rdchem
from rdkit.Geometry import rdGeometry
from rdkit.Chem.inchi import *
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdCIPLabeler import *
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdmolops import *
def CanonSmiles(smi, useChiral=1):
    m = MolFromSmiles(smi)
    return MolToSmiles(m, useChiral)