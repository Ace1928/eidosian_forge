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
def CollectResults(indices, dataSet, composite, callback=None, appendExamples=0, errorEstimate=0):
    """ screens a set of examples through a composite and returns the
  results
#DOC

  **Arguments**

    - examples: the examples to be screened (a sequence of sequences)
       it's assumed that the last element in each example is it's "value"

    - composite:  the composite model to be used

    - callback: (optional)  if provided, this should be a function
      taking a single argument that is called after each example is
      screened with the number of examples screened so far as the
      argument.

    - appendExamples: (optional)  this value is passed on to the
      composite's _ClassifyExample()_ method.

    - errorEstimate: (optional) calculate the "out of bag" error
      estimate for the composite using Breiman's definition.  This
      only makes sense when screening the original data set!
      [L. Breiman "Out-of-bag Estimation", UC Berkeley Dept of
      Statistics Technical Report (1996)]

  **Returns**

    a list of 3-tuples _nExamples_ long:

      1)  answer: the value from the example

      2)  pred: the composite model's prediction

      3)  conf: the confidence of the composite

  """
    for j in range(len(composite)):
        tmp = composite.GetModel(j)
        if hasattr(tmp, '_trainIndices') and type(tmp._trainIndices) != dict:
            tis = {}
            if hasattr(tmp, '_trainIndices'):
                for v in tmp._trainIndices:
                    tis[v] = 1
            tmp._trainIndices = tis
    nPts = len(indices)
    res = [None] * nPts
    for i in range(nPts):
        idx = indices[i]
        example = dataSet[idx]
        if errorEstimate:
            use = []
            for j in range(len(composite)):
                mdl = composite.GetModel(j)
                if not mdl._trainIndices.get(idx, 0):
                    use.append(j)
        else:
            use = None
        pred, conf = composite.ClassifyExample(example, appendExample=appendExamples, onlyModels=use)
        if composite.GetActivityQuantBounds():
            answer = composite.QuantizeActivity(example)[-1]
        else:
            answer = example[-1]
        res[i] = (answer, pred, conf)
        if callback:
            callback(i)
    return res