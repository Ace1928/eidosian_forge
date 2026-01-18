import os
import numpy
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
def _ReadPatts(fileName):
    """ *Internal Use Only*

    parses the pattern list from the data file

  """
    patts = {}
    order = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != '#':
            splitLine = line.split('\t')
            if len(splitLine) >= 4 and splitLine[0] != '':
                sma = splitLine[1]
                if sma != 'SMARTS':
                    sma.replace('"', '')
                    p = Chem.MolFromSmarts(sma)
                    if p:
                        if len(splitLine[0]) > 1 and splitLine[0][1] not in 'S0123456789':
                            cha = splitLine[0][:2]
                        else:
                            cha = splitLine[0][0]
                        logP = float(splitLine[2])
                        if splitLine[3] != '':
                            mr = float(splitLine[3])
                        else:
                            mr = 0.0
                        if cha not in order:
                            order.append(cha)
                        l = patts.get(cha, [])
                        l.append((sma, p, logP, mr))
                        patts[cha] = l
                else:
                    print('Problems parsing smarts: %s' % sma)
    return (order, patts)