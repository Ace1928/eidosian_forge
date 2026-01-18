import random
from rdkit import RDRandom
def SplitIndices(nPts, frac, silent=1, legacy=0, replacement=0):
    """ splits a set of indices into a data set into 2 pieces

    **Arguments**

     - nPts: the total number of points

     - frac: the fraction of the data to be put in the first data set

     - silent: (optional) toggles display of stats

     - legacy: (optional) use the legacy splitting approach

     - replacement: (optional) use selection with replacement

   **Returns**

     a 2-tuple containing the two sets of indices.

   **Notes**

     - the _legacy_ splitting approach uses randomly-generated floats
       and compares them to _frac_.  This is provided for
       backwards-compatibility reasons.

     - the default splitting approach uses a random permutation of
       indices which is split into two parts.

     - selection with replacement can generate duplicates.


  **Usage**:

  We'll start with a set of indices and pick from them using
  the three different approaches:
  >>> from rdkit.ML.Data import DataUtils

  The base approach always returns the same number of compounds in
  each set and has no duplicates:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> test,train = SplitIndices(10,.5)
  >>> test
  [1, 5, 6, 4, 2]
  >>> train
  [3, 0, 7, 8, 9]

  >>> test,train = SplitIndices(10,.5)
  >>> test
  [5, 2, 9, 8, 7]
  >>> train
  [6, 0, 3, 1, 4]


  The legacy approach can return varying numbers, but still has no
  duplicates.  Note the indices come back ordered:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> test,train = SplitIndices(10,.5,legacy=1)
  >>> test
  [3, 5, 7, 8, 9]
  >>> train
  [0, 1, 2, 4, 6]

  >>> test,train = SplitIndices(10,.5,legacy=1)
  >>> test
  [0, 1, 2, 3, 5, 8, 9]
  >>> train
  [4, 6, 7]

  The replacement approach returns a fixed number in the training set,
  a variable number in the test set and can contain duplicates in the
  training set.
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> test,train = SplitIndices(10,.5,replacement=1)
  >>> test
  [9, 9, 8, 0, 5]
  >>> train
  [1, 2, 3, 4, 6, 7]
  >>> test,train = SplitIndices(10,.5,replacement=1)
  >>> test
  [4, 5, 1, 1, 4]
  >>> train
  [0, 2, 3, 6, 7, 8, 9]

  """
    if frac < 0.0 or frac > 1.0:
        raise ValueError('frac must be between 0.0 and 1.0 (frac=%f)' % frac)
    if replacement:
        nTrain = int(nPts * frac)
        resData = [None] * nTrain
        resTest = []
        for i in range(nTrain):
            val = int(RDRandom.random() * nPts)
            if val == nPts:
                val = nPts - 1
            resData[i] = val
        for i in range(nPts):
            if i not in resData:
                resTest.append(i)
    elif legacy:
        resData = []
        resTest = []
        for i in range(nPts):
            val = RDRandom.random()
            if val < frac:
                resData.append(i)
            else:
                resTest.append(i)
    else:
        perm = list(range(nPts))
        RDRandom.shuffle(perm, random=random.random)
        nTrain = int(nPts * frac)
        resData = list(perm[:nTrain])
        resTest = list(perm[nTrain:])
    if not silent:
        print('Training with %d (of %d) points.' % (len(resData), nPts))
        print('\t%d points are in the hold-out set.' % len(resTest))
    return (resData, resTest)