import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def ScoreFromLists(bitLists, suppl, catalog, maxPts=-1, actName='', acts=None, nActs=2, reportFreq=10):
    """  similar to _ScoreMolecules()_, but uses pre-calculated bit lists
      for the molecules (this speeds things up a lot)


      **Arguments**

        - bitLists: sequence of on bit sequences for the input molecules

        - suppl: the input supplier (we read activities from here)

        - catalog: the FragmentCatalog

        - maxPts: (optional) the maximum number of molecules to be
          considered

        - actName: (optional) the name of the molecule's activity property.
          If this is not provided, the molecule's last property will be used.

        - nActs: (optional) number of possible activity values

        - reportFreq: (optional) how often to display status information

      **Returns**

         the results table (a 3D array of ints nBits x 2 x nActs)

    """
    nBits = catalog.GetFPLength()
    if maxPts > 0:
        nPts = maxPts
    else:
        nPts = len(bitLists)
    resTbl = numpy.zeros((nBits, 2, nActs), numpy.int32)
    if not actName and (not acts):
        actName = suppl[0].GetPropNames()[-1]
    suppl.reset()
    for i in range(1, nPts + 1):
        mol = next(suppl)
        if not acts:
            act = int(mol.GetProp(actName))
        else:
            act = acts[i - 1]
        if i and (not i % reportFreq):
            message('Done %d of %d\n' % (i, nPts))
        ids = set()
        for id_ in bitLists[i - 1]:
            ids.add(id_ - 1)
        for j in range(nBits):
            resTbl[j, 0, act] += 1
        for id_ in ids:
            resTbl[id_, 0, act] -= 1
            resTbl[id_, 1, act] += 1
    return resTbl