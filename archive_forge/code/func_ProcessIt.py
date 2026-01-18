from warnings import warn
import pickle
import sys
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import ScreenComposite
from rdkit.ML.Data import Stats
from rdkit.ML.DecTree import Tree, TreeUtils
def ProcessIt(composites, nToConsider=3, verbose=0):
    composite = composites[0]
    nComposites = len(composites)
    ns = composite.GetDescriptorNames()
    if len(ns) > 2:
        globalRes = {}
        nDone = 1
        descNames = {}
        for composite in composites:
            if verbose > 0:
                print('#------------------------------------')
                print('Doing: ', nDone)
            nModels = len(composite)
            nDone += 1
            res = {}
            for i in range(len(composite)):
                model = composite.GetModel(i)
                if isinstance(model, Tree.TreeNode):
                    levels = TreeUtils.CollectLabelLevels(model, {}, 0, nToConsider)
                    TreeUtils.CollectDescriptorNames(model, descNames, 0, nToConsider)
                    for descId in levels.keys():
                        v = res.get(descId, numpy.zeros(nToConsider, float))
                        v[levels[descId]] += 1.0 / nModels
                        res[descId] = v
            for k in res:
                v = globalRes.get(k, numpy.zeros(nToConsider, float))
                v += res[k] / nComposites
                globalRes[k] = v
            if verbose > 0:
                for k in res.keys():
                    name = descNames[k]
                    strRes = ', '.join(['%4.2f' % x for x in res[k]])
                    print('%s,%s,%5.4f' % (name, strRes, sum(res[k])))
                print()
        if verbose >= 0:
            print('# Average Descriptor Positions')
        retVal = []
        for k in globalRes:
            name = descNames[k]
            if verbose >= 0:
                strRes = ', '.join(['%4.2f' % x for x in globalRes[k]])
                print('%s,%s,%5.4f' % (name, strRes, sum(globalRes[k])))
            tmp = [name]
            tmp.extend(globalRes[k])
            tmp.append(sum(globalRes[k]))
            retVal.append(tmp)
        if verbose >= 0:
            print()
    else:
        retVal = []
    return retVal