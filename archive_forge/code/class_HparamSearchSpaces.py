from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HparamSearchSpaces(_messages.Message):
    """Hyperparameter search spaces. These should be a subset of
  training_options.

  Fields:
    activationFn: Activation functions of neural network models.
    batchSize: Mini batch sample size.
    boosterType: Booster type for boosted tree models.
    colsampleBylevel: Subsample ratio of columns for each level for boosted
      tree models.
    colsampleBynode: Subsample ratio of columns for each node(split) for
      boosted tree models.
    colsampleBytree: Subsample ratio of columns when constructing each tree
      for boosted tree models.
    dartNormalizeType: Dart normalization type for boosted tree models.
    dropout: Dropout probability for dnn model training and boosted tree
      models using dart booster.
    hiddenUnits: Hidden units for neural network models.
    l1Reg: L1 regularization coefficient.
    l2Reg: L2 regularization coefficient.
    learnRate: Learning rate of training jobs.
    maxTreeDepth: Maximum depth of a tree for boosted tree models.
    minSplitLoss: Minimum split loss for boosted tree models.
    minTreeChildWeight: Minimum sum of instance weight needed in a child for
      boosted tree models.
    numClusters: Number of clusters for k-means.
    numFactors: Number of latent factors to train on.
    numParallelTree: Number of parallel trees for boosted tree models.
    optimizer: Optimizer of TF models.
    subsample: Subsample the training data to grow tree to prevent overfitting
      for boosted tree models.
    treeMethod: Tree construction algorithm for boosted tree models.
    walsAlpha: Hyperparameter for matrix factoration when implicit feedback
      type is specified.
  """
    activationFn = _messages.MessageField('StringHparamSearchSpace', 1)
    batchSize = _messages.MessageField('IntHparamSearchSpace', 2)
    boosterType = _messages.MessageField('StringHparamSearchSpace', 3)
    colsampleBylevel = _messages.MessageField('DoubleHparamSearchSpace', 4)
    colsampleBynode = _messages.MessageField('DoubleHparamSearchSpace', 5)
    colsampleBytree = _messages.MessageField('DoubleHparamSearchSpace', 6)
    dartNormalizeType = _messages.MessageField('StringHparamSearchSpace', 7)
    dropout = _messages.MessageField('DoubleHparamSearchSpace', 8)
    hiddenUnits = _messages.MessageField('IntArrayHparamSearchSpace', 9)
    l1Reg = _messages.MessageField('DoubleHparamSearchSpace', 10)
    l2Reg = _messages.MessageField('DoubleHparamSearchSpace', 11)
    learnRate = _messages.MessageField('DoubleHparamSearchSpace', 12)
    maxTreeDepth = _messages.MessageField('IntHparamSearchSpace', 13)
    minSplitLoss = _messages.MessageField('DoubleHparamSearchSpace', 14)
    minTreeChildWeight = _messages.MessageField('IntHparamSearchSpace', 15)
    numClusters = _messages.MessageField('IntHparamSearchSpace', 16)
    numFactors = _messages.MessageField('IntHparamSearchSpace', 17)
    numParallelTree = _messages.MessageField('IntHparamSearchSpace', 18)
    optimizer = _messages.MessageField('StringHparamSearchSpace', 19)
    subsample = _messages.MessageField('DoubleHparamSearchSpace', 20)
    treeMethod = _messages.MessageField('StringHparamSearchSpace', 21)
    walsAlpha = _messages.MessageField('DoubleHparamSearchSpace', 22)