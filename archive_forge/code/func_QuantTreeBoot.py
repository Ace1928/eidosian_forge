import numpy
from rdkit import RDRandom as random
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3, QuantTree
from rdkit.ML.InfoTheory import entropy
def QuantTreeBoot(examples, attrs, nPossibleVals, nBoundsPerVar, initialVar=None, maxDepth=-1, **kwargs):
    """ Bootstrapping code for the QuantTree

      If _initialVar_ is not set, the algorithm will automatically
       choose the first variable in the tree (the standard greedy
       approach).  Otherwise, _initialVar_ will be used as the first
       split.

    """
    attrs = list(attrs)
    for i in range(len(nBoundsPerVar)):
        if nBoundsPerVar[i] == -1 and i in attrs:
            attrs.remove(i)
    tree = QuantTree.QuantTreeNode(None, 'node')
    nPossibleRes = nPossibleVals[-1]
    tree._nResultCodes = nPossibleRes
    resCodes = [int(x[-1]) for x in examples]
    counts = [0] * nPossibleRes
    for res in resCodes:
        counts[res] += 1
    if initialVar is None:
        best, gainHere, qBounds = FindBest(resCodes, examples, nBoundsPerVar, nPossibleRes, nPossibleVals, attrs, **kwargs)
    else:
        best = initialVar
        if nBoundsPerVar[best] > 0:
            vTable = map(lambda x, z=best: x[z], examples)
            qBounds, gainHere = Quantize.FindVarMultQuantBounds(vTable, nBoundsPerVar[best], resCodes, nPossibleRes)
        elif nBoundsPerVar[best] == 0:
            vTable = ID3.GenVarTable(examples, nPossibleVals, [best])[0]
            gainHere = entropy.InfoGain(vTable)
            qBounds = []
        else:
            gainHere = -1000000.0
            qBounds = []
    tree.SetName('Var: %d' % best)
    tree.SetData(gainHere)
    tree.SetLabel(best)
    tree.SetTerminal(0)
    tree.SetQuantBounds(qBounds)
    nextAttrs = list(attrs)
    if not kwargs.get('recycleVars', 0):
        nextAttrs.remove(best)
    indices = list(range(len(examples)))
    if len(qBounds) > 0:
        for bound in qBounds:
            nextExamples = []
            for index in list(indices):
                ex = examples[index]
                if ex[best] < bound:
                    nextExamples.append(ex)
                    indices.remove(index)
            if len(nextExamples):
                tree.AddChildNode(BuildQuantTree(nextExamples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=1, maxDepth=maxDepth, **kwargs))
            else:
                v = numpy.argmax(counts)
                tree.AddChild('%d??' % v, label=v, data=0.0, isTerminal=1)
        nextExamples = []
        for index in indices:
            nextExamples.append(examples[index])
        if len(nextExamples) != 0:
            tree.AddChildNode(BuildQuantTree(nextExamples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=1, maxDepth=maxDepth, **kwargs))
        else:
            v = numpy.argmax(counts)
            tree.AddChild('%d??' % v, label=v, data=0.0, isTerminal=1)
    else:
        for val in range(nPossibleVals[best]):
            nextExamples = []
            for example in examples:
                if example[best] == val:
                    nextExamples.append(example)
            if len(nextExamples) != 0:
                tree.AddChildNode(BuildQuantTree(nextExamples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=1, maxDepth=maxDepth, **kwargs))
            else:
                v = numpy.argmax(counts)
                tree.AddChild('%d??' % v, label=v, data=0.0, isTerminal=1)
    return tree