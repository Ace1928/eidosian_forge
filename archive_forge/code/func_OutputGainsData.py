import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def OutputGainsData(outF, gains, cat, nActs=2):
    actHeaders = ['Act-%d' % x for x in range(nActs)]
    if cat:
        outF.write('id,Description,Gain,%s\n' % ','.join(actHeaders))
    else:
        outF.write('id,Gain,%s\n' % ','.join(actHeaders))
    for entry in gains:
        id_ = int(entry[0])
        outL = [str(id_)]
        if cat:
            descr = cat.GetBitDescription(id_)
            outL.append(descr)
        outL.append('%.6f' % entry[1])
        outL += ['%d' % x for x in entry[2:]]
        outF.write(','.join(outL))
        outF.write('\n')