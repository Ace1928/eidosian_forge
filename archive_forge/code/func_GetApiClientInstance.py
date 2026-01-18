from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import functools
import json
import re
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import operations
from googlecloudsdk.api_lib.functions.v2 import util as v2_util
from googlecloudsdk.api_lib.storage import storage_api as gcs_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.generated_clients.apis.cloudfunctions.v1 import cloudfunctions_v1_messages
import six.moves.http_client
def GetApiClientInstance(track=calliope_base.ReleaseTrack.GA):
    """Returns the GCFv1 client instance."""
    endpoint_override = v2_util.GetApiEndpointOverride()
    if not endpoint_override or 'autopush-cloudfunctions' not in endpoint_override:
        return apis.GetClientInstance(_API_NAME, _GetApiVersion(track))
    log.info('Temporarily overriding cloudfunctions endpoint to staging-cloudfunctions.sandbox.googleapis.com so that GCFv1 autopush resources can be accessed.')
    properties.VALUES.api_endpoint_overrides.Property('cloudfunctions').Set('https://staging-cloudfunctions.sandbox.googleapis.com/')
    client = apis.GetClientInstance(_API_NAME, _GetApiVersion(track))
    properties.VALUES.api_endpoint_overrides.Property('cloudfunctions').Set('https://autopush-cloudfunctions.sandbox.googleapis.com/')
    return client