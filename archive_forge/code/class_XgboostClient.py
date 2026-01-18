import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
class XgboostClient(PredictionClient):
    """A loaded xgboost model to be used for prediction."""

    def __init__(self, booster):
        self._booster = booster

    def predict(self, inputs, stats=None, **kwargs):
        stats = stats or prediction_utils.Stats()
        stats[prediction_utils.FRAMEWORK] = prediction_utils.XGBOOST_FRAMEWORK_NAME
        stats[prediction_utils.ENGINE] = prediction_utils.XGBOOST_FRAMEWORK_NAME
        import xgboost as xgb
        try:
            inputs_dmatrix = xgb.DMatrix(inputs)
        except Exception as e:
            logging.exception('Could not initialize DMatrix from inputs.')
            raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL, 'Could not initialize DMatrix from inputs: ' + str(e))
        with stats.time(prediction_utils.SESSION_RUN_TIME):
            try:
                return self._booster.predict(inputs_dmatrix, **kwargs)
            except Exception as e:
                logging.exception('Exception during predicting with xgboost model: ')
                raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL, 'Exception during xgboost prediction: ' + str(e))