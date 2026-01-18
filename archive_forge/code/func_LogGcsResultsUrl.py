from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def LogGcsResultsUrl(self):
    log.status.Print('Raw results will be stored in your GCS bucket at [{0}]\n'.format(self._gcs_results_url))