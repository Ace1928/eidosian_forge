from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _CheckServiceFlag(self, service, required=False):
    """Raises an exception that lists the parameters that can't be changed."""
    disable_service_flags = self.type_metadata.disable_service_flags
    if disable_service_flags and service:
        raise exceptions.ArgumentError('--service not allowed for integration type [{}]'.format(self.type_metadata.integration_type))
    if not disable_service_flags and (not service) and required:
        raise exceptions.ArgumentError('--service is required')