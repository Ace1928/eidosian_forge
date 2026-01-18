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
def LoadSDF(filename, idName='ID', molColName='ROMol', includeFingerprints=False, isomericSmiles=True, smilesName=None, embedProps=False, removeHs=True, strictParsing=True, sanitize=True):
    """Read file in SDF format and return as Pandas data frame.
      If embedProps=True all properties also get embedded in Mol objects in the molecule column.
      If molColName=None molecules would not be present in resulting DataFrame (only properties
      would be read).
      
      Sanitize boolean is passed on to Chem.ForwardSDMolSupplier sanitize. 
      If neither molColName nor smilesName are set, sanitize=false.
      """
    if isinstance(filename, str):
        if filename.lower()[-3:] == '.gz':
            import gzip
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        close = f.close
    else:
        f = filename
        close = None
    records = []
    indices = []
    if molColName is None and smilesName is None:
        sanitize = False
    for i, mol in enumerate(Chem.ForwardSDMolSupplier(f, sanitize=sanitize, removeHs=removeHs, strictParsing=strictParsing)):
        if mol is None:
            continue
        row = dict(((k, mol.GetProp(k)) for k in mol.GetPropNames()))
        if molColName is not None and (not embedProps):
            for prop in mol.GetPropNames():
                mol.ClearProp(prop)
        if mol.HasProp('_Name'):
            row[idName] = mol.GetProp('_Name')
        if smilesName is not None:
            try:
                row[smilesName] = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
            except Exception:
                log.warning('No valid smiles could be generated for molecule %s', i)
                row[smilesName] = None
        if molColName is not None and (not includeFingerprints):
            row[molColName] = mol
        elif molColName is not None:
            row[molColName] = _MolPlusFingerprint(mol)
        records.append(row)
        indices.append(i)
    if close is not None:
        close()
    df = pd.DataFrame(records, index=indices)
    ChangeMoleculeRendering(df)
    return df