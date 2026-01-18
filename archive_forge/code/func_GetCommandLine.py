from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.ML import CompositeRun, ScreenComposite
from rdkit.ML.Composite import BayesComposite, Composite
from rdkit.ML.Data import DataUtils, SplitData
from rdkit.utils import listutils
def GetCommandLine(details):
    """ #DOC

  """
    args = ['BuildComposite']
    args.append('-n %d' % details.nModels)
    if details.filterFrac != 0.0:
        args.append('-F %.3f -v %d' % (details.filterFrac, details.filterVal))
    if details.modelFilterFrac != 0.0:
        args.append('--modelFiltFrac=%.3f --modelFiltVal=%d' % (details.modelFilterFrac, details.modelFilterVal))
    if details.splitRun:
        args.append('-s -f %.3f' % details.splitFrac)
    if details.shuffleActivities:
        args.append('-S')
    if details.randomActivities:
        args.append('-r')
    if details.threshold > 0.0:
        args.append('-t %.3f' % details.threshold)
    if details.activityBounds:
        args.append('-Q "%s"' % details.activityBoundsVals)
    if details.dbName:
        args.append('-d %s' % details.dbName)
    if details.detailedRes:
        args.append('-D')
    if hasattr(details, 'noScreen') and details.noScreen:
        args.append('--noScreen')
    if details.persistTblName and details.dbName:
        args.append('-p %s' % details.persistTblName)
    if details.note:
        args.append('-N %s' % details.note)
    if details.useTrees:
        if details.limitDepth > 0:
            args.append('-L %d' % details.limitDepth)
        if details.lessGreedy:
            args.append('-g')
        if details.qBounds:
            shortBounds = listutils.CompactListRepr(details.qBounds)
            if details.qBounds:
                args.append('-q "%s"' % shortBounds)
        elif details.qBounds:
            args.append('-q "%s"' % details.qBoundCount)
        if details.pruneIt:
            args.append('--prune')
        if details.startAt:
            args.append('-G %d' % details.startAt)
        if details.recycleVars:
            args.append('--recycle')
        if details.randomDescriptors:
            args.append('--randomDescriptors=%d' % details.randomDescriptors)
    if details.useSigTrees:
        args.append('--doSigTree')
        if details.limitDepth > 0:
            args.append('-L %d' % details.limitDepth)
        if details.randomDescriptors:
            args.append('--randomDescriptors=%d' % details.randomDescriptors)
    if details.useKNN:
        args.append('--doKnn --knnK %d' % details.knnNeighs)
        if details.knnDistFunc == 'Tanimoto':
            args.append('--knnTanimoto')
        else:
            args.append('--knnEuclid')
    if details.useNaiveBayes:
        args.append('--doNaiveBayes')
        if details.mEstimateVal >= 0.0:
            args.append('--mEstimateVal=%.3f' % details.mEstimateVal)
    if details.replacementSelection:
        args.append('--replacementSelection')
    if details.tableName:
        args.append(details.tableName)
    return ' '.join(args)