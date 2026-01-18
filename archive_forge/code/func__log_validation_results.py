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
def _log_validation_results(self):
    """Output collected validation results."""
    log.out.Print(json.dumps(self._VALIDATION_RESULTS))