from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import exceptions as dataflow_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
Get region to be used in Dataflow services.

  Args:
    args: Argument passed in when running gcloud dataflow command

  Returns:
    Region specified by user from --region flag in args, then fall back to
    'us-central1'.
  