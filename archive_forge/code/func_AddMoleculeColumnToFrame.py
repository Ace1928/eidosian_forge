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
def AddMoleculeColumnToFrame(frame, smilesCol='Smiles', molCol='ROMol', includeFingerprints=False):
    """Converts the molecules contains in "smilesCol" to RDKit molecules and appends them to the
    dataframe "frame" using the specified column name.
    If desired, a fingerprint can be computed and stored with the molecule objects to accelerate
    substructure matching
    """
    if not includeFingerprints:
        frame[molCol] = frame[smilesCol].map(Chem.MolFromSmiles)
    else:
        frame[molCol] = frame[smilesCol].map(lambda smiles: _MolPlusFingerprint(Chem.MolFromSmiles(smiles)))
    ChangeMoleculeRendering(frame)