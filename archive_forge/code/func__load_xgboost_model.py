import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
def _load_xgboost_model(model_path):
    """Loads an xgboost model from GCS or local.

  Args:
      model_path: path to the directory containing the xgboost model.bst file.
        This path can be either a local path or a GCS path.

  Returns:
    A xgboost.Booster with the model at model_path loaded.

  Raises:
    PredictionError: If there is a problem while loading the file.
  """
    import xgboost as xgb
    if model_path.startswith('gs://'):
        prediction_utils.copy_model_to_local(model_path, prediction_utils.LOCAL_MODEL_PATH)
        model_path = prediction_utils.LOCAL_MODEL_PATH
    model_file = os.path.join(model_path, MODEL_FILE_NAME_BST)
    if not os.path.exists(model_file):
        return None
    try:
        return xgb.Booster(model_file=model_file)
    except xgb.core.XGBoostError as e:
        error_msg = 'Could not load the model: {}.'.format(os.path.join(model_path, MODEL_FILE_NAME_BST))
        logging.exception(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, '{}. {}.'.format(error_msg, str(e)))