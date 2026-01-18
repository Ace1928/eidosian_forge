import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def FilterData(inData, val, frac, col=-1, indicesToUse=None, indicesOnly=0):
    """
  #DOC
    """
    if frac < 0 or frac > 1:
        raise ValueError('filter fraction out of bounds')
    try:
        inData[0][col]
    except IndexError:
        raise ValueError('target column index out of range')
    if indicesToUse:
        tmp = [inData[x] for x in indicesToUse]
    else:
        tmp = list(inData)
    nOrig = len(tmp)
    sortOrder = list(range(nOrig))
    sortOrder.sort(key=lambda x: tmp[x][col])
    tmp = [tmp[x] for x in sortOrder]
    start = 0
    while start < nOrig and tmp[start][col] != val:
        start += 1
    if start >= nOrig:
        raise ValueError('target value (%d) not found in data' % val)
    finish = start + 1
    while finish < nOrig and tmp[finish][col] == val:
        finish += 1
    nWithVal = finish - start
    nOthers = len(tmp) - nWithVal
    currFrac = float(nWithVal) / nOrig
    if currFrac < frac:
        nTgtFinal = nWithVal
        nFinal = int(round(nWithVal / frac))
        nOthersFinal = nFinal - nTgtFinal
        while float(nTgtFinal) / nFinal > frac:
            nTgtFinal -= 1
            nFinal -= 1
    else:
        nOthersFinal = nOthers
        nFinal = int(round(nOthers / (1 - frac)))
        nTgtFinal = nFinal - nOthersFinal
        while float(nTgtFinal) / nFinal < frac:
            nOthersFinal -= 1
            nFinal -= 1
    others = list(range(start)) + list(range(finish, nOrig))
    othersTake = permutation(nOthers)
    others = [others[x] for x in othersTake[:nOthersFinal]]
    targets = list(range(start, finish))
    targetsTake = permutation(nWithVal)
    targets = [targets[x] for x in targetsTake[:nTgtFinal]]
    indicesToKeep = targets + others
    res = []
    rej = []
    if not indicesOnly:
        for i in permutation(nOrig):
            if i in indicesToKeep:
                res.append(tmp[i])
            else:
                rej.append(tmp[i])
    else:
        for i in permutation(nOrig):
            if not indicesToUse:
                idx = sortOrder[i]
            else:
                idx = indicesToUse[sortOrder[i]]
            if i in indicesToKeep:
                res.append(idx)
            else:
                rej.append(idx)
    return (res, rej)