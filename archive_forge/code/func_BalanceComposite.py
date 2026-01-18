import copy
import numpy
def BalanceComposite(model, set1, set2, weight, targetSize, names1=None, names2=None):
    """ adjusts the contents of the composite model so as to maximize
    the weighted classification accuracty across the two data sets.

    The resulting composite model, with _targetSize_ models, is returned.

    **Notes**:

      - if _names1_ and _names2_ are not provided, _set1_ and _set2_ should
        have the same ordering of columns and _model_ should have already
        have had _SetInputOrder()_ called.

  """
    S1 = len(set1)
    S2 = len(set2)
    weight1 = float(S1 + S2) * (1 - weight) / S1
    weight2 = float(S1 + S2) * weight / S2
    res = copy.copy(model)
    res.modelList = []
    res.errList = []
    res.countList = []
    res.quantizationRequirements = []
    startSize = len(model)
    scores = numpy.zeros(startSize, float)
    actQuantBounds = model.GetActivityQuantBounds()
    if names1 is not None:
        model.SetInputOrder(names1)
    for pt in set1:
        pred, conf = model.ClassifyExample(pt)
        if actQuantBounds:
            ans = model.QuantizeActivity(pt)[-1]
        else:
            ans = pt[-1]
        votes = model.GetVoteDetails()
        for i in range(startSize):
            if votes[i] == ans:
                scores[i] += weight1
    if names2 is not None:
        model.SetInputOrder(names2)
    for pt in set2:
        pred, conf = model.ClassifyExample(pt)
        if actQuantBounds:
            ans = model.QuantizeActivity(pt)[-1]
        else:
            ans = pt[-1]
        votes = model.GetVoteDetails()
        for i in range(startSize):
            if votes[i] == ans:
                scores[i] += weight2
    nPts = S1 + S2
    scores /= nPts
    bestOrder = list(numpy.argsort(scores))
    bestOrder.reverse()
    print('\tTAKE:', bestOrder[:targetSize])
    for i in range(targetSize):
        idx = bestOrder[i]
        mdl = model.modelList[idx]
        res.modelList.append(mdl)
        res.errList.append(1.0 - scores[idx])
        res.countList.append(1)
        res.quantizationRequirements.append(0)
    return res