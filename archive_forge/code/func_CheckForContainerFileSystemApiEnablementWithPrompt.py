from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def CheckForContainerFileSystemApiEnablementWithPrompt(project):
    """Checks if the Container File System API is enabled."""
    service_name = 'containerfilesystem.googleapis.com'
    try:
        if not enable_api.IsServiceEnabled(project, service_name):
            log.warning('Container File System API (containerfilesystem.googleapis.com) has not been enabled on the project. Please enable it for image streaming to fully work. For additional details, please refer to https://cloud.google.com/kubernetes-engine/docs/how-to/image-streaming#requirements')
    except (exceptions.GetServicePermissionDeniedException, apitools_exceptions.HttpError):
        log.warning('Failed to check if Container File System API (containerfilesystem.googleapis.com) has been enabled. Please make sure to enable it for image streaming to work. For additional details, please refer to https://cloud.google.com/kubernetes-engine/docs/how-to/image-streaming#requirements')