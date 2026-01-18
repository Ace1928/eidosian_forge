from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.core import exceptions as gcloud_core_exceptions
Custom error class for Audit Manager related exceptions.

  Attributes:
    http_exception: core http exception thrown by gcloud
    suggested_command_purpose: what the suggested command achieves
    suggested_command: suggested command to help fix the exception
  