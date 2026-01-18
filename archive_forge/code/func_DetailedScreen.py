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
def DetailedScreen(indices, data, composite, threshold=0, screenResults=None, goodVotes=None, badVotes=None, noVotes=None, callback=None, appendExamples=0, errorEstimate=0):
    """ screens a set of examples cross a composite and breaks the
      predictions into *correct*,*incorrect* and *unclassified* sets.
#DOC
  **Arguments**

    - examples: the examples to be screened (a sequence of sequences)
       it's assumed that the last element in each example is its "value"

    - composite:  the composite model to be used

    - threshold: (optional) the threshold to be used to decide whether
      or not a given prediction should be kept

    - screenResults: (optional) the results of screening the results
      (a sequence of 3-tuples in the format returned by
      _CollectResults()_).  If this is provided, the examples will not
      be screened again.

    - goodVotes,badVotes,noVotes: (optional)  if provided these should
      be lists (or anything supporting an _append()_ method) which
      will be used to pass the screening results back.

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

  **Notes**

    - since this function doesn't return anything, if one or more of
      the arguments _goodVotes_, _badVotes_, and _noVotes_ is not
      provided, there's not much reason to call it

  """
    if screenResults is None:
        screenResults = CollectResults(indices, data, composite, callback=callback, appendExamples=appendExamples, errorEstimate=errorEstimate)
    if goodVotes is None:
        goodVotes = []
    if badVotes is None:
        badVotes = []
    if noVotes is None:
        noVotes = []
    for i in range(len(screenResults)):
        answer, pred, conf = screenResults[i]
        if conf > threshold:
            if pred != answer:
                badVotes.append((answer, pred, conf, i))
            else:
                goodVotes.append((answer, pred, conf, i))
        else:
            noVotes.append((answer, pred, conf, i))