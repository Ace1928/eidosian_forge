from rdkit.ML.Data import SplitData
from rdkit.ML.KNN import DistFunctions
from rdkit.ML.KNN.KNNClassificationModel import KNNClassificationModel
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
def CrossValidate(knnMod, testExamples, appendExamples=0):
    """
  Determines the classification error for the testExamples

  **Arguments**

    - tree: a decision tree (or anything supporting a _ClassifyExample()_ method)

    - testExamples: a list of examples to be used for testing

    - appendExamples: a toggle which is passed along to the tree as it does
      the classification. The trees can use this to store the examples they
      classify locally.

  **Returns**

    a 2-tuple consisting of:
      """
    nTest = len(testExamples)
    if isinstance(knnMod, KNNClassificationModel):
        badExamples = []
        nBad = 0
        for i in range(nTest):
            testEx = testExamples[i]
            trueRes = testEx[-1]
            res = knnMod.ClassifyExample(testEx, appendExamples)
            if trueRes != res:
                badExamples.append(testEx)
                nBad += 1
        return (float(nBad) / nTest, badExamples)
    elif isinstance(knnMod, KNNRegressionModel):
        devSum = 0.0
        for i in range(nTest):
            testEx = testExamples[i]
            trueRes = testEx[-1]
            res = knnMod.PredictExample(testEx, appendExamples)
            devSum += abs(trueRes - res)
        return (devSum / nTest, None)
    raise ValueError('Unrecognized Model Type')