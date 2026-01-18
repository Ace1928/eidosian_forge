from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1AblationAttribution(_messages.Message):
    """Attributes credit to model inputs by ablating features (ie. setting them
  to their default/missing values) and computing corresponding model score
  delta per feature. The term "ablation" is in reference to running an
  "ablation study" to analyze input effects on the outcome of interest, which
  in this case is the model's output. This attribution method is supported for
  TensorFlow and XGBoost models.

  Fields:
    numFeatureInteractions: Number of feature interactions to account for in
      the ablation process, capped at the maximum number of provided input
      features. Currently, only the value 1 is supported.
  """
    numFeatureInteractions = _messages.IntegerField(1, variant=_messages.Variant.INT32)