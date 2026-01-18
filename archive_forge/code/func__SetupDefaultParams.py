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
def _SetupDefaultParams(self, user_provided_params):
    """Ensures that default parameters have a value if not set."""
    for param in self.type_metadata.parameters:
        if param.default and param.name not in user_provided_params:
            user_provided_params[param.name] = param.default