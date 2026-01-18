import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
class SessionClient(TensorFlowClient):
    """A client for Prediction that uses Session.run."""

    def __init__(self, session, signature_map):
        self._session = session
        super(SessionClient, self).__init__(signature_map)

    def predict(self, inputs, stats=None, signature_name=None, **unused_kwargs):
        """Produces predictions for the given inputs.

    Args:
      inputs: a dict mapping input names to values
      stats: Stats object for recording timing information.
      signature_name: name of SignatureDef to use in this prediction
      **unused_kwargs: placeholder, pre/postprocess may have additional args

    Returns:
      A dict mapping output names to output values, similar to the input
      dict.
    """
        stats = stats or prediction_utils.Stats()
        stats[prediction_utils.ENGINE] = 'SessionRun'
        stats[prediction_utils.FRAMEWORK] = prediction_utils.TENSORFLOW_FRAMEWORK_NAME
        with stats.time(prediction_utils.UNALIAS_TIME):
            _, signature = self.get_signature(signature_name)
            fetches = [output.name for output in signature.outputs.values()]
            try:
                unaliased = {signature.inputs[key].name: val for key, val in six.iteritems(inputs)}
            except Exception as e:
                logging.exception('Input mismatch.')
                raise PredictionError(PredictionError.INVALID_INPUTS, 'Input mismatch: ' + str(e))
        with stats.time(prediction_utils.SESSION_RUN_TIME):
            try:
                outputs = self._session.run(fetches=fetches, feed_dict=unaliased)
            except Exception as e:
                logging.exception('Exception running the graph.')
                raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL, 'Exception during running the graph: ' + str(e))
        with stats.time(prediction_utils.ALIAS_TIME):
            return dict(zip(six.iterkeys(signature.outputs), outputs))