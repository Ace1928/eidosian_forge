from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import BuildComposite, CompositeRun, ScreenComposite
from rdkit.ML.Composite import AdjustComposite
from rdkit.ML.Data import DataUtils, SplitData
def GrowIt(details, composite, progressCallback=None, saveIt=1, setDescNames=0, data=None):
    """ does the actual work of building a composite model

    **Arguments**

      - details:  a _CompositeRun.CompositeRun_ object containing details
        (options, parameters, etc.) about the run

      - composite: the composite model to grow

      - progressCallback: (optional) a function which is called with a single
        argument (the number of models built so far) after each model is built.

      - saveIt: (optional) if this is nonzero, the resulting model will be pickled
        and dumped to the filename specified in _details.outName_

      - setDescNames: (optional) if nonzero, the composite's _SetInputOrder()_ method
        will be called using the results of the data set's _GetVarNames()_ method;
        it is assumed that the details object has a _descNames attribute which
        is passed to the composites _SetDescriptorNames()_ method.  Otherwise
        (the default), _SetDescriptorNames()_ gets the results of _GetVarNames()_.

      - data: (optional) the data set to be used.  If this is not provided, the
        data set described in details will be used.

    **Returns**

      the enlarged composite model


  """
    details.rundate = time.asctime()
    if data is None:
        fName = details.tableName.strip()
        if details.outName == '':
            details.outName = fName + '.pkl'
        if details.dbName == '':
            data = DataUtils.BuildQuantDataSet(fName)
        elif details.qBounds != []:
            details.tableName = fName
            data = details.GetDataSet()
        else:
            data = DataUtils.DBToQuantData(details.dbName, fName, quantName=details.qTableName, user=details.dbUser, password=details.dbPassword)
    seed = composite._randomSeed
    DataUtils.InitRandomNumbers(seed)
    if details.shuffleActivities == 1:
        DataUtils.RandomizeActivities(data, shuffle=1, runDetails=details)
    elif details.randomActivities == 1:
        DataUtils.RandomizeActivities(data, shuffle=0, runDetails=details)
    namedExamples = data.GetNamedData()
    trainExamples = namedExamples
    nExamples = len(trainExamples)
    message('Training with %d examples' % nExamples)
    message('\t%d descriptors' % (len(trainExamples[0]) - 2))
    nVars = data.GetNVars()
    nPossibleVals = composite.nPossibleVals
    attrs = list(range(1, nVars + 1))
    if details.useTrees:
        from rdkit.ML.DecTree import CrossValidate, PruneTree
        if details.qBounds != []:
            from rdkit.ML.DecTree import BuildQuantTree
            builder = BuildQuantTree.QuantTreeBoot
        else:
            from rdkit.ML.DecTree import ID3
            builder = ID3.ID3Boot
        driver = CrossValidate.CrossValidationDriver
        pruner = PruneTree.PruneTree
        if setDescNames:
            composite.SetInputOrder(data.GetVarNames())
        composite.Grow(trainExamples, attrs, [0] + nPossibleVals, buildDriver=driver, pruner=pruner, nTries=details.nModels, pruneIt=details.pruneIt, lessGreedy=details.lessGreedy, needsQuantization=0, treeBuilder=builder, nQuantBounds=details.qBounds, startAt=details.startAt, maxDepth=details.limitDepth, progressCallback=progressCallback, silent=not _verbose)
    else:
        from rdkit.ML.Neural import CrossValidate
        driver = CrossValidate.CrossValidationDriver
        composite.Grow(trainExamples, attrs, [0] + nPossibleVals, nTries=details.nModels, buildDriver=driver, needsQuantization=0)
    composite.AverageErrors()
    composite.SortModels()
    modelList, counts, avgErrs = composite.GetAllData()
    counts = numpy.array(counts)
    avgErrs = numpy.array(avgErrs)
    composite._varNames = data.GetVarNames()
    for i in range(len(modelList)):
        modelList[i].NameModel(composite._varNames)
    weightedErrs = counts * avgErrs
    averageErr = sum(weightedErrs) / sum(counts)
    devs = avgErrs - averageErr
    devs = devs * counts
    devs = numpy.sqrt(devs * devs)
    avgDev = sum(devs) / sum(counts)
    if _verbose:
        message('# Overall Average Error: %%% 5.2f, Average Deviation: %%% 6.2f' % (100.0 * averageErr, 100.0 * avgDev))
    if details.bayesModel:
        composite.Train(trainExamples, verbose=0)
    badExamples = []
    if not details.detailedRes:
        if _verbose:
            message('Testing all examples')
        wrong = BuildComposite.testall(composite, namedExamples, badExamples)
        if _verbose:
            message('%d examples (%% %5.2f) were misclassified' % (len(wrong), 100.0 * float(len(wrong)) / float(len(namedExamples))))
        _runDetails.overall_error = float(len(wrong)) / len(namedExamples)
    if details.detailedRes:
        if _verbose:
            message('\nEntire data set:')
        resTup = ScreenComposite.ShowVoteResults(range(data.GetNPts()), data, composite, nPossibleVals[-1], details.threshold)
        nGood, nBad, _, avgGood, avgBad, _, voteTab = resTup
        nPts = len(namedExamples)
        nClass = nGood + nBad
        _runDetails.overall_error = float(nBad) / nClass
        _runDetails.overall_correct_conf = avgGood
        _runDetails.overall_incorrect_conf = avgBad
        _runDetails.overall_result_matrix = repr(voteTab)
        nRej = nClass - nPts
        if nRej > 0:
            _runDetails.overall_fraction_dropped = float(nRej) / nPts
    return composite