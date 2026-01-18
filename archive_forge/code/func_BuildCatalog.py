import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def BuildCatalog(suppl, maxPts=-1, groupFileName=None, minPath=2, maxPath=6, reportFreq=10):
    """ builds a fragment catalog from a set of molecules in a delimited text block

      **Arguments**

        - suppl: a mol supplier

        - maxPts: (optional) if provided, this will set an upper bound on the
          number of points to be considered

        - groupFileName: (optional) name of the file containing functional group
          information

        - minPath, maxPath: (optional) names of the minimum and maximum path lengths
          to be considered

        - reportFreq: (optional) how often to display status information

      **Returns**

        a FragmentCatalog

    """
    if groupFileName is None:
        groupFileName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fpParams = FragmentCatalog.FragCatParams(minPath, maxPath, groupFileName)
    catalog = FragmentCatalog.FragCatalog(fpParams)
    fgen = FragmentCatalog.FragCatGenerator()
    if maxPts > 0:
        nPts = maxPts
    elif hasattr(suppl, '__len__'):
        nPts = len(suppl)
    else:
        nPts = -1
    for i, mol in enumerate(suppl):
        if i == nPts:
            break
        if i and (not i % reportFreq):
            if nPts > -1:
                message('Done %d of %d, %d paths\n' % (i, nPts, catalog.GetFPLength()))
            else:
                message('Done %d, %d paths\n' % (i, catalog.GetFPLength()))
        fgen.AddFragsFromMol(mol, catalog)
    return catalog