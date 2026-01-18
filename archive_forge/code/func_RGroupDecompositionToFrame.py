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
def RGroupDecompositionToFrame(groups, mols, include_core=False, redraw_sidechains=False):
    """ returns a dataframe with the results of R-Group Decomposition

    >>> from rdkit import Chem
    >>> from rdkit.Chem import rdRGroupDecomposition
    >>> from rdkit.Chem import PandasTools
    >>> import pandas as pd
    >>> scaffold = Chem.MolFromSmiles('c1ccccn1')
    >>> mols = [Chem.MolFromSmiles(smi) for smi in 'c1c(F)cccn1 c1c(Cl)c(C)ccn1 c1c(O)cccn1 c1c(F)c(C)ccn1 c1cc(Cl)c(F)cn1'.split()]
    >>> groups,_ = rdRGroupDecomposition.RGroupDecompose([scaffold],mols,asSmiles=False,asRows=False)
    >>> df = PandasTools.RGroupDecompositionToFrame(groups,mols,include_core=False)
    >>> list(df.columns)
    ['Mol', 'R1', 'R2']
    >>> df = PandasTools.RGroupDecompositionToFrame(groups,mols,include_core=True)
    >>> list(df.columns)
    ['Mol', 'Core', 'R1', 'R2']
    >>> len(df)
    5
    >>> df.columns() # doctest: +SKIP
    <class 'pandas*...*DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 4 columns):
    Mol     5 non-null object
    Core    5 non-null object
    R1      5 non-null object
    R2      5 non-null object
    dtypes: object(4)
    memory usage: *...*

    """
    groups = groups.copy()
    cols = ['Mol'] + list(groups.keys())
    if redraw_sidechains:
        from rdkit.Chem import rdDepictor
        for k, vl in groups.items():
            if k == 'Core':
                continue
            for i, v in enumerate(vl):
                vl[i] = Chem.RemoveHs(v)
                rdDepictor.Compute2DCoords(vl[i])
    if not include_core:
        cols.remove('Core')
        del groups['Core']
    groups['Mol'] = mols
    frame = pd.DataFrame(groups, columns=cols)
    ChangeMoleculeRendering(frame)
    return frame