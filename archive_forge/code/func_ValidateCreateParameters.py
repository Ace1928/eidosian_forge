from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def ValidateCreateParameters(integration_type, parameters, service):
    """Validates given params conform to what's expected from the integration."""
    types_utils.CheckValidIntegrationType(integration_type)
    validate = validator.GetIntegrationValidator(integration_type)
    validate.ValidateCreateParameters(parameters, service)