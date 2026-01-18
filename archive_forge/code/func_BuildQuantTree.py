import numpy
from rdkit import RDRandom as random
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3, QuantTree
from rdkit.ML.InfoTheory import entropy
def BuildQuantTree(examples, target, attrs, nPossibleVals, nBoundsPerVar, depth=0, maxDepth=-1, exIndices=None, **kwargs):
    """
      **Arguments**

        - examples: a list of lists (nInstances x nVariables+1) of variable
          values + instance values

        - target: an int

        - attrs: a list of ints indicating which variables can be used in the tree

        - nPossibleVals: a list containing the number of possible values of
                     every variable.

        - nBoundsPerVar: the number of bounds to include for each variable

        - depth: (optional) the current depth in the tree

        - maxDepth: (optional) the maximum depth to which the tree
                     will be grown
      **Returns**

       a QuantTree.QuantTreeNode with the decision tree

      **NOTE:** This code cannot bootstrap (start from nothing...)
            use _QuantTreeBoot_ (below) for that.
    """
    tree = QuantTree.QuantTreeNode(None, 'node')
    tree.SetData(-666)
    nPossibleRes = nPossibleVals[-1]
    if exIndices is None:
        exIndices = list(range(len(examples)))
    resCodes = [int(x[-1]) for x in (examples[y] for y in exIndices)]
    counts = [0] * nPossibleRes
    for res in resCodes:
        counts[res] += 1
    nzCounts = numpy.nonzero(counts)[0]
    if len(nzCounts) == 1:
        res = nzCounts[0]
        tree.SetLabel(res)
        tree.SetName(str(res))
        tree.SetTerminal(1)
    elif len(attrs) == 0 or (maxDepth >= 0 and depth > maxDepth):
        v = numpy.argmax(counts)
        tree.SetLabel(v)
        tree.SetName('%d?' % v)
        tree.SetTerminal(1)
    else:
        best, _, bestBounds = FindBest(resCodes, examples, nBoundsPerVar, nPossibleRes, nPossibleVals, attrs, exIndices=exIndices, **kwargs)
        nextAttrs = attrs[:]
        if not kwargs.get('recycleVars', 0):
            nextAttrs.remove(best)
        tree.SetName('Var: %d' % best)
        tree.SetLabel(best)
        tree.SetQuantBounds(bestBounds)
        tree.SetTerminal(0)
        indices = exIndices[:]
        if len(bestBounds) > 0:
            for bound in bestBounds:
                nextExamples = []
                for index in indices[:]:
                    ex = examples[index]
                    if ex[best] < bound:
                        nextExamples.append(index)
                        indices.remove(index)
                if len(nextExamples) == 0:
                    v = numpy.argmax(counts)
                    tree.AddChild('%d' % v, label=v, data=0.0, isTerminal=1)
                else:
                    tree.AddChildNode(BuildQuantTree(examples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=depth + 1, maxDepth=maxDepth, exIndices=nextExamples, **kwargs))
            nextExamples = []
            for index in indices:
                nextExamples.append(index)
            if len(nextExamples) == 0:
                v = numpy.argmax(counts)
                tree.AddChild('%d' % v, label=v, data=0.0, isTerminal=1)
            else:
                tree.AddChildNode(BuildQuantTree(examples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=depth + 1, maxDepth=maxDepth, exIndices=nextExamples, **kwargs))
        else:
            for val in range(nPossibleVals[best]):
                nextExamples = []
                for idx in exIndices:
                    if examples[idx][best] == val:
                        nextExamples.append(idx)
                if len(nextExamples) == 0:
                    v = numpy.argmax(counts)
                    tree.AddChild('%d' % v, label=v, data=0.0, isTerminal=1)
                else:
                    tree.AddChildNode(BuildQuantTree(examples, best, nextAttrs, nPossibleVals, nBoundsPerVar, depth=depth + 1, maxDepth=maxDepth, exIndices=nextExamples, **kwargs))
    return tree