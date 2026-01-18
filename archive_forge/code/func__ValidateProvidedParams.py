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
def _ValidateProvidedParams(self, user_provided_params):
    """Checks that the user provided parameters exist in the mapping."""
    invalid_params = []
    allowed_params = [param.name for param in self.type_metadata.parameters]
    for param in user_provided_params:
        if param not in allowed_params:
            invalid_params.append(param)
    if invalid_params:
        raise exceptions.ArgumentError('The following parameters: {} are not allowed'.format(self._RemoveEncoding(invalid_params)))