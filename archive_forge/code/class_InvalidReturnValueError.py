from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from googlecloudsdk.command_lib.ml_engine import local_predict
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
class InvalidReturnValueError(core_exceptions.Error):
    """Indicates that the return value from local_predict has some error."""
    pass