import argparse
import os
import sys
import warnings
from absl import app
import tensorflow as tf  # pylint: disable=unused-import
from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.lite.toco.logging import gen_html
from tensorflow.python import tf2
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.util import keras_deps
class _ParseBooleanFlag(argparse.Action):
    """Helper class to parse boolean flag that optionally accepts truth value."""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs != '?':
            raise ValueError("This parser only supports nargs='?' (0 or 1 additional arguments)")
        super(_ParseBooleanFlag, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            flag_value = True
        elif values.lower() == 'true':
            flag_value = True
        elif values.lower() == 'false':
            flag_value = False
        else:
            raise ValueError('Invalid argument to --{}. Must use flag alone, or specify true/false.'.format(self.dest))
        setattr(namespace, self.dest, flag_value)