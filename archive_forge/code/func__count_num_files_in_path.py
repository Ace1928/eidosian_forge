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
def _count_num_files_in_path(model_path, specified_file_names):
    """Count how many specified files exist in model_path.

  Args:
    model_path: The local path to the directory that contains the model file.
    specified_file_names: The file names to be checked

  Returns:
    An integer indicating how many specified_file_names are found in model_path.
  """
    num_matches = 0
    for file_name in specified_file_names:
        if os.path.exists(os.path.join(model_path, file_name)):
            num_matches += 1
    return num_matches