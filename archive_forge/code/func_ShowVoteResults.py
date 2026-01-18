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
def ShowVoteResults(indices, data, composite, nResultCodes, threshold, verbose=1, screenResults=None, callback=None, appendExamples=0, goodVotes=None, badVotes=None, noVotes=None, errorEstimate=0):
    """ screens the results and shows a detailed workup

  The work of doing the screening and processing the results is
  handled by _DetailedScreen()_
#DOC

  **Arguments**

    - examples: the examples to be screened (a sequence of sequences)
       it's assumed that the last element in each example is its "value"

    - composite:  the composite model to be used

    - nResultCodes: the number of possible results the composite can
      return

    - threshold: the threshold to be used to decide whether or not a
      given prediction should be kept

    - screenResults: (optional) the results of screening the results
      (a sequence of 3-tuples in the format returned by
      _CollectResults()_).  If this is provided, the examples will not
      be screened again.

    - callback: (optional)  if provided, this should be a function
      taking a single argument that is called after each example is
      screened with the number of examples screened so far as the
      argument.

    - appendExamples: (optional)  this value is passed on to the
      composite's _ClassifyExample()_ method.

    - goodVotes,badVotes,noVotes: (optional)  if provided these should
      be lists (or anything supporting an _append()_ method) which
      will be used to pass the screening results back.

    - errorEstimate: (optional) calculate the "out of bag" error
      estimate for the composite using Breiman's definition.  This
      only makes sense when screening the original data set!
      [L. Breiman "Out-of-bag Estimation", UC Berkeley Dept of
      Statistics Technical Report (1996)]

  **Returns**

    a 7-tuple:

      1) the number of good (correct) predictions

      2) the number of bad (incorrect) predictions

      3) the number of predictions skipped due to the _threshold_

      4) the average confidence in the good predictions

      5) the average confidence in the bad predictions

      6) the average confidence in the skipped predictions

      7) the results table

  """
    nExamples = len(indices)
    if goodVotes is None:
        goodVotes = []
    if badVotes is None:
        badVotes = []
    if noVotes is None:
        noVotes = []
    DetailedScreen(indices, data, composite, threshold, screenResults=screenResults, goodVotes=goodVotes, badVotes=badVotes, noVotes=noVotes, callback=callback, appendExamples=appendExamples, errorEstimate=errorEstimate)
    nBad = len(badVotes)
    nGood = len(goodVotes)
    nClassified = nGood + nBad
    if verbose:
        print('\n\t*** Vote Results ***')
        print('misclassified: %d/%d (%%%4.2f)\t%d/%d (%%%4.2f)' % (nBad, nExamples, 100.0 * float(nBad) / nExamples, nBad, nClassified, 100.0 * float(nBad) / nClassified))
    nSkip = len(noVotes)
    if nSkip > 0:
        if verbose:
            print('skipped: %d/%d (%%% 4.2f)' % (nSkip, nExamples, 100.0 * float(nSkip) / nExamples))
        noConf = numpy.array([x[2] for x in noVotes])
        avgSkip = sum(noConf) / float(nSkip)
    else:
        avgSkip = 0.0
    if nBad > 0:
        badConf = numpy.array([x[2] for x in badVotes])
        avgBad = sum(badConf) / float(nBad)
    else:
        avgBad = 0.0
    if nGood > 0:
        goodRes = [x[1] for x in goodVotes]
        goodConf = numpy.array([x[2] for x in goodVotes])
        avgGood = sum(goodConf) / float(nGood)
    else:
        goodRes = []
        goodConf = []
        avgGood = 0.0
    if verbose:
        print()
        print('average correct confidence:   % 6.4f' % avgGood)
        print('average incorrect confidence: % 6.4f' % avgBad)
    voteTab = numpy.zeros((nResultCodes, nResultCodes), numpy.int32)
    for res in goodRes:
        voteTab[res, res] += 1
    for ans, res, conf, idx in badVotes:
        voteTab[ans, res] += 1
    if verbose:
        print()
        print('\tResults Table:')
        vTab = voteTab.transpose()
        colCounts = numpy.sum(vTab, 0)
        rowCounts = numpy.sum(vTab, 1)
        message('')
        for i in range(nResultCodes):
            if rowCounts[i] == 0:
                rowCounts[i] = 1
            row = vTab[i]
            message('    ', noRet=1)
            for j in range(nResultCodes):
                entry = row[j]
                message(' % 6d' % entry, noRet=1)
            message('     | % 4.2f' % (100.0 * vTab[i, i] / rowCounts[i]))
        message('    ', noRet=1)
        for i in range(nResultCodes):
            message('-------', noRet=1)
        message('')
        message('    ', noRet=1)
        for i in range(nResultCodes):
            if colCounts[i] == 0:
                colCounts[i] = 1
            message(' % 6.2f' % (100.0 * vTab[i, i] / colCounts[i]), noRet=1)
        message('')
    return (nGood, nBad, nSkip, avgGood, avgBad, avgSkip, voteTab)