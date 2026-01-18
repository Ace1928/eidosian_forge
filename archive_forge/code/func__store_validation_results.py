from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import copy
import json
import shlex
from googlecloudsdk import gcloud_main
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _store_validation_results(self, success, command_string, error_message=None, error_type=None):
    """Store information related to the command validation."""
    validation_output = copy.deepcopy(_PARSING_OUTPUT_TEMPLATE)
    validation_output['command_string'] = command_string
    validation_output['success'] = success
    validation_output['error_message'] = error_message
    validation_output['error_type'] = error_type
    self._VALIDATION_RESULTS.append(validation_output)