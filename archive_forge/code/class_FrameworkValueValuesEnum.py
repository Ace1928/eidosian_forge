from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FrameworkValueValuesEnum(_messages.Enum):
    """Optional. The machine learning framework AI Platform uses to train
    this version of the model. Valid values are `TENSORFLOW`, `SCIKIT_LEARN`,
    `XGBOOST`. If you do not specify a framework, AI Platform will analyze
    files in the deployment_uri to determine a framework. If you choose
    `SCIKIT_LEARN` or `XGBOOST`, you must also set the runtime version of the
    model to 1.4 or greater. Do **not** specify a framework if you're
    deploying a [custom prediction routine](/ai-
    platform/prediction/docs/custom-prediction-routines) or if you're using a
    [custom container](/ai-platform/prediction/docs/use-custom-container).

    Values:
      FRAMEWORK_UNSPECIFIED: Unspecified framework. Assigns a value based on
        the file suffix.
      TENSORFLOW: Tensorflow framework.
      SCIKIT_LEARN: Scikit-learn framework.
      XGBOOST: XGBoost framework.
    """
    FRAMEWORK_UNSPECIFIED = 0
    TENSORFLOW = 1
    SCIKIT_LEARN = 2
    XGBOOST = 3