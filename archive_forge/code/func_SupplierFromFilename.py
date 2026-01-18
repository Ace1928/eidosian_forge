from rdkit import DataStructs, RDConfig, rdBase
from rdkit.Chem import rdchem
from rdkit.Geometry import rdGeometry
from rdkit.Chem.inchi import *
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdCIPLabeler import *
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdmolops import *
def SupplierFromFilename(fileN, delim='', **kwargs):
    ext = fileN.split('.')[-1].lower()
    if ext == 'sdf':
        suppl = SDMolSupplier(fileN, **kwargs)
    elif ext == 'csv':
        if not delim:
            delim = ','
        suppl = SmilesMolSupplier(fileN, delimiter=delim, **kwargs)
    elif ext == 'txt':
        if not delim:
            delim = '\t'
        suppl = SmilesMolSupplier(fileN, delimiter=delim, **kwargs)
    elif ext == 'tdt':
        suppl = TDTMolSupplier(fileN, delimiter=delim, **kwargs)
    else:
        raise ValueError('unrecognized extension: %s' % ext)
    return suppl