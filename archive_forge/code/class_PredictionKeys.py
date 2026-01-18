from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
class PredictionKeys(object):
    """Enum for canonical model prediction keys.

  The following values are defined:
  PREDICTIONS: Used by models that predict values, such as regressor models.
  """
    CLASSES = 'classes'
    CLASS_IDS = 'class_ids'
    ALL_CLASSES = 'all_classes'
    ALL_CLASS_IDS = 'all_class_ids'
    LOGISTIC = 'logistic'
    LOGITS = 'logits'
    PREDICTIONS = 'predictions'
    PROBABILITIES = 'probabilities'
    TOP_K = 'top_k'