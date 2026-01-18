import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def AlignMol(mol, scaffold):
    """
    Aligns mol (RDKit mol object) to scaffold (SMILES string)
    """
    scaffold = Chem.MolFromSmiles(scaffold)
    AllChem.Compute2DCoords(scaffold)
    AllChem.GenerateDepictionMatching2DStructure(mol, scaffold)
    return mol