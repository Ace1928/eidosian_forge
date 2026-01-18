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
class TensorFlowClient(PredictionClient):
    """A client for Prediction that uses Session.run."""

    def __init__(self, signature_map, *args, **kwargs):
        self._signature_map = signature_map
        super(TensorFlowClient, self).__init__(*args, **kwargs)

    @property
    def signature_map(self):
        return self._signature_map

    def get_signature(self, signature_name=None):
        """Gets tensorflow signature for the given signature_name.

    Args:
      signature_name: string The signature name to use to choose the signature
                      from the signature map.

    Returns:
      a pair of signature_name and signature. The first element is the
      signature name in string that is actually used. The second one is the
      signature.

    Raises:
      PredictionError: when the signature is not found with the given signature
      name or when there are more than one signatures in the signature map.
    """
        if not signature_name and len(self.signature_map) == 1:
            return (list(self.signature_map.keys())[0], list(self.signature_map.values())[0])
        key = signature_name or DEFAULT_SERVING_SIGNATURE_DEF_KEY
        if key in self.signature_map:
            return (key, self.signature_map[key])
        else:
            raise PredictionError(PredictionError.INVALID_INPUTS, 'No signature found for signature key %s.' % signature_name)