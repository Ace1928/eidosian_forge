from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
Lists information about the supported locations for this service.

      Args:
        request: (CloudkmsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      