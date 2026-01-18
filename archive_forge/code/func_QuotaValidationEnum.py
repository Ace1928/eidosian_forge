from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def QuotaValidationEnum(quota_validation):
    """Checks if a quota validation provided by user is valid and returns corresponding enum type.

  Args:
    quota_validation: value for quota validation.

  Returns:
    quota validation enum
  Raises:
    ArgumentTypeError: If the value provided by user is not valid.
  """
    messages = configmanager_util.GetMessagesModule()
    quota_validation_enum_dict = {'QUOTA_VALIDATION_UNSPECIFIED': messages.Deployment.QuotaValidationValueValuesEnum.QUOTA_VALIDATION_UNSPECIFIED, 'ENABLED': messages.Deployment.QuotaValidationValueValuesEnum.ENABLED, 'ENFORCED': messages.Deployment.QuotaValidationValueValuesEnum.ENFORCED}
    if quota_validation is None:
        return
    if quota_validation not in quota_validation_enum_dict:
        raise arg_parsers.ArgumentTypeError("quota validation does not support: '{0}', supported values are: {1}".format(quota_validation, list(quota_validation_enum_dict)))
    return quota_validation_enum_dict[quota_validation]