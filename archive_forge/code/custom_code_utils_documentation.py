import inspect
import json
import os
import pydoc  # used for importing python classes from their FQN
import sys
from ._interfaces import Model
from .prediction_utils import PredictionError
Validates a user provided implementation of Model class.

  Args:
    user_class: The user provided custom Model class.

  Raises:
    PredictionError: for any of the following:
      (1) the user model class does not have the correct method signatures for
      the predict method
  