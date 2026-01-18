from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _GenerateLabels(labels, messages, resource):
    """Parses input file or values and returns a list of additional properties.

  Args:
    labels: User-defined metadata for the deployment.
    messages: ModuleType, the messages module that lets us form blueprints API
      messages based on our protos.
    resource: Resource type, can be deployment or preview.

  Returns:
    The additional_properties list.
  """
    labels_message = {}
    if labels is not None:
        if resource == 'deployment':
            labels_message = messages.Deployment.LabelsValue(additionalProperties=[messages.Deployment.LabelsValue.AdditionalProperty(key=key, value=value) for key, value in six.iteritems(labels)])
        elif resource == 'preview':
            labels_message = messages.Preview.LabelsValue(additionalProperties=[messages.Preview.LabelsValue.AdditionalProperty(key=key, value=value) for key, value in six.iteritems(labels)])
        return labels_message