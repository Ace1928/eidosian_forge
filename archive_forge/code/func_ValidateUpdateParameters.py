from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def ValidateUpdateParameters(integration_type, parameters):
    """Validate params for update commands for a given integration."""
    types_utils.CheckValidIntegrationType(integration_type)
    validate = validator.GetIntegrationValidator(integration_type)
    validate.ValidateUpdateParameters(parameters)