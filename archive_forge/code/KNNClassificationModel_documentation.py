from rdkit.ML.KNN import KNNModel
 Classify a an example by looking at its closest neighbors

    The class assigned to this example is same as the class for the mojority of its
    _k neighbors

    **Arguments**

    - examples: the example to be classified

    - appendExamples: if this is nonzero then the example will be stored on this model

    - neighborList: if provided, will be used to return the list of neighbors

    **Returns**

      - the classification of _example_
    