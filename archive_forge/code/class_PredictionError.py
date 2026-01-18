import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
class PredictionError(Exception):
    """Customer exception for known prediction exception."""
    FAILED_TO_LOAD_MODEL = PredictionErrorType(message='Failed to load model', code=0)
    INVALID_INPUTS = PredictionErrorType('Invalid inputs', code=1)
    FAILED_TO_RUN_MODEL = PredictionErrorType(message='Failed to run the provided model', code=2)
    INVALID_OUTPUTS = PredictionErrorType(message='There was a problem processing the outputs', code=3)
    INVALID_USER_CODE = PredictionErrorType(message='There was a problem processing the user code', code=4)
    FAILED_TO_ACCESS_METADATA_SERVER = PredictionErrorType(message='Could not get an access token from the metadata server', code=5)

    @property
    def error_code(self):
        return self.args[0].code

    @property
    def error_message(self):
        return self.args[0].message

    @property
    def error_detail(self):
        return self.args[1]

    def __str__(self):
        return '%s: %s (Error code: %d)' % (self.error_message, self.error_detail, self.error_code)