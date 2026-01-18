from rdkit.ML.Data import SplitData
from rdkit.ML.KNN import DistFunctions
from rdkit.ML.KNN.KNNClassificationModel import KNNClassificationModel
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
 Driver function for building a KNN model of a specified type

  **Arguments**

    - examples: the full set of examples

    - numNeigh: number of neighbors for the KNN model (basically k in k-NN)

    - knnModel: the type of KNN model (a classification vs regression model)

    - holdOutFrac: the fraction of the data which should be reserved for the hold-out set
      (used to calculate error)

    - silent: a toggle used to control how much visual noise this makes as it goes

    - calcTotalError: a toggle used to indicate whether the classification error
      of the tree should be calculated using the entire data set (when true) or just
      the training hold out set (when false)
      