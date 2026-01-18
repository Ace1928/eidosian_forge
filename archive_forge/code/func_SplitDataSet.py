import random
from rdkit import RDRandom
def SplitDataSet(data, frac, silent=0):
    """ splits a data set into two pieces

    **Arguments**

     - data: a list of examples to be split

     - frac: the fraction of the data to be put in the first data set

     - silent: controls the amount of visual noise produced.

   **Returns**

     a 2-tuple containing the two new data sets.

  """
    if frac < 0.0 or frac > 1.0:
        raise ValueError('frac must be between 0.0 and 1.0')
    nOrig = len(data)
    train, test = SplitIndices(nOrig, frac, silent=1)
    resData = [data[x] for x in train]
    resTest = [data[x] for x in test]
    if not silent:
        print('Training with %d (of %d) points.' % (len(resData), nOrig))
        print('\t%d points are in the hold-out set.' % len(resTest))
    return (resData, resTest)