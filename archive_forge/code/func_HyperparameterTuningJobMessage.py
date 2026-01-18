from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import yaml
def HyperparameterTuningJobMessage(self):
    """Returns the HyperparameterTuningJob resource message."""
    return self._GetMessage('HyperparameterTuningJob')