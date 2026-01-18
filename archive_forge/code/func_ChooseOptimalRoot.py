import numpy
from rdkit.ML.Data import SplitData
from rdkit.ML.DecTree import ID3, randomtest
def ChooseOptimalRoot(examples, trainExamples, testExamples, attrs, nPossibleVals, treeBuilder, nQuantBounds=[], **kwargs):
    """ loops through all possible tree roots and chooses the one which produces the best tree

  **Arguments**

    - examples: the full set of examples

    - trainExamples: the training examples

    - testExamples: the testing examples

    - attrs: a list of attributes to consider in the tree building

    - nPossibleVals: a list of the number of possible values each variable can adopt

    - treeBuilder: the function to be used to actually build the tree

    - nQuantBounds: an optional list.  If present, it's assumed that the builder
      algorithm takes this argument as well (for building QuantTrees)

  **Returns**

    The best tree found

  **Notes**

    1) Trees are built using _trainExamples_

    2) Testing of each tree (to determine which is best) is done using _CrossValidate_ and
       the entire set of data (i.e. all of _examples_)

    3) _trainExamples_ is not used at all, which immediately raises the question of
       why it's even being passed in

  """
    attrs = attrs[:]
    if nQuantBounds:
        for i in range(len(nQuantBounds)):
            if nQuantBounds[i] == -1 and i in attrs:
                attrs.remove(i)
    nAttrs = len(attrs)
    trees = [None] * nAttrs
    errs = [0] * nAttrs
    errs[0] = 1000000.0
    for i in range(1, nAttrs):
        argD = {'initialVar': attrs[i]}
        argD.update(kwargs)
        if nQuantBounds is None or nQuantBounds == []:
            trees[i] = treeBuilder(trainExamples, attrs, nPossibleVals, **argD)
        else:
            trees[i] = treeBuilder(trainExamples, attrs, nPossibleVals, nQuantBounds, **argD)
        if trees[i]:
            errs[i], _ = CrossValidate(trees[i], examples, appendExamples=0)
        else:
            errs[i] = 1000000.0
    best = numpy.argmin(errs)
    return trees[best]