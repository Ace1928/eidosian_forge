import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _translate_errors(func):

    def func_wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except FileNotFoundError as e:
            raise errors.NotFoundError(None, None, str(e))
    return func_wrapper