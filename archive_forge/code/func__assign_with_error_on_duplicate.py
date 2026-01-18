from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import boto3
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
from six.moves import configparser
def _assign_with_error_on_duplicate(key, value, result_dict):
    """Assigns value to results_dict and raises error on duplicate key."""
    if key in result_dict:
        raise KeyError('Duplicate key in file: {}'.format(key))
    result_dict[key] = value