from warnings import warn
import os
import pickle
import sys
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils, SplitData
def GetScreenImage(nGood, nBad, nRej, size=None):
    if not hasPil:
        return None
    try:
        nTot = float(nGood) + float(nBad) + float(nRej)
    except TypeError:
        nGood = nGood[0]
        nBad = nBad[0]
        nRej = nRej[0]
        nTot = float(nGood) + float(nBad) + float(nRej)
    if not nTot:
        return None
    goodColor = (100, 100, 255)
    badColor = (255, 100, 100)
    rejColor = (255, 255, 100)
    pctGood = float(nGood) / nTot
    pctBad = float(nBad) / nTot
    pctRej = float(nRej) / nTot
    if size is None:
        size = (100, 100)
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    box = (0, 0, size[0] - 1, size[1] - 1)
    startP = -90
    endP = int(startP + pctGood * 360)
    draw.pieslice(box, startP, endP, fill=goodColor)
    startP = endP
    endP = int(startP + pctBad * 360)
    draw.pieslice(box, startP, endP, fill=badColor)
    startP = endP
    endP = int(startP + pctRej * 360)
    draw.pieslice(box, startP, endP, fill=rejColor)
    return img