from rdkit.ML.KNN import KNNModel
 Generates a prediction for an example by looking at its closest neighbors

    **Arguments**

      - examples: the example to be classified

      - appendExamples: if this is nonzero then the example will be stored on this model

      - weightedAverage: if provided, the neighbors' contributions to the value will be
                         weighed by their reciprocal square distance

      - neighborList: if provided, will be used to return the list of neighbors

    **Returns**

      - the classification of _example_

    