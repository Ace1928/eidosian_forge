from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.command_lib.util.args import map_util
import six
def EnvVarValueType(value):
    if not isinstance(value, six.text_type):
        raise argparse.ArgumentTypeError('Environment variable values must be strings. Found {} (type {})'.format(value, type(value)))
    return value