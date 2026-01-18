import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def ScoreMolecules(suppl, catalog, maxPts=-1, actName='', acts=None, nActs=2, reportFreq=10):
    """ scores the compounds in a supplier using a catalog

      **Arguments**

        - suppl: a mol supplier

        - catalog: the FragmentCatalog

        - maxPts: (optional) the maximum number of molecules to be
          considered

        - actName: (optional) the name of the molecule's activity property.
          If this is not provided, the molecule's last property will be used.

        - acts: (optional) a sequence of activity values (integers).
          If not provided, the activities will be read from the molecules.

        - nActs: (optional) number of possible activity values

        - reportFreq: (optional) how often to display status information

      **Returns**

        a 2-tuple:

          1) the results table (a 3D array of ints nBits x 2 x nActs)

          2) a list containing the on bit lists for each molecule

    """
    nBits = catalog.GetFPLength()
    resTbl = numpy.zeros((nBits, 2, nActs), numpy.int32)
    obls = []
    if not actName and (not acts):
        actName = suppl[0].GetPropNames()[-1]
    fpgen = FragmentCatalog.FragFPGenerator()
    suppl.reset()
    i = 1
    for mol in suppl:
        if i and (not i % reportFreq):
            message('Done %d.\n' % i)
        if mol:
            if not acts:
                act = int(mol.GetProp(actName))
            else:
                act = acts[i - 1]
            fp = fpgen.GetFPForMol(mol, catalog)
            obls.append([x for x in fp.GetOnBits()])
            for j in range(nBits):
                resTbl[j, 0, act] += 1
            for id_ in obls[i - 1]:
                resTbl[id_ - 1, 0, act] -= 1
                resTbl[id_ - 1, 1, act] += 1
        else:
            obls.append([])
        i += 1
    return (resTbl, obls)