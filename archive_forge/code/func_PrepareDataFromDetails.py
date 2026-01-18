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
def PrepareDataFromDetails(model, details, data, verbose=0):
    if hasattr(details, 'doHoldout') and details.doHoldout or (hasattr(details, 'doTraining') and details.doTraining):
        try:
            splitF = model._splitFrac
        except AttributeError:
            pass
        else:
            if verbose:
                message('s', noRet=1)
            if hasattr(details, 'errorEstimate') and details.errorEstimate and hasattr(details, 'doHoldout') and details.doHoldout:
                message('*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*')
                message('******  WARNING: OOB screening should not be combined with doHoldout option.')
                message('*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*')
            trainIdx, testIdx = SplitData.SplitIndices(data.GetNPts(), splitF, silent=1)
        if hasattr(details, 'filterFrac') and details.filterFrac != 0.0:
            if verbose:
                message('f', noRet=1)
            trainFilt, temp = DataUtils.FilterData(data, details.filterVal, details.filterFrac, -1, indicesToUse=trainIdx, indicesOnly=1)
            testIdx += temp
            trainIdx = trainFilt
    elif hasattr(details, 'errorEstimate') and details.errorEstimate:
        if hasattr(details, 'filterFrac') and details.filterFrac != 0.0:
            if verbose:
                message('f', noRet=1)
            testIdx, trainIdx = DataUtils.FilterData(data, details.filterVal, details.filterFrac, -1, indicesToUse=range(data.GetNPts()), indicesOnly=1)
            testIdx.extend(trainIdx)
        else:
            testIdx = list(range(data.GetNPts()))
        trainIdx = []
    else:
        testIdx = list(range(data.GetNPts()))
        trainIdx = []
    if hasattr(details, 'doTraining') and details.doTraining:
        testIdx, trainIdx = (trainIdx, testIdx)
    return (trainIdx, testIdx)