from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.binauthz import apis
def EnsureEnabledFalseIsShown(cv_config):
    """Ensures that "enabled" is shown when printing ContinuousValidationConfig.

  Explicitly sets ContinuousValidationConfig.enforcementPolicyConfig.enabled
  to False when it's unset, so the field is printed as "enabled: false",
  instead of omitting the "enabled" key when CV is not enabled.

  Args:
    cv_config: A ContinuousValidationConfig.

  Returns:
    The modified cv_config.
  """
    if not cv_config.enforcementPolicyConfig or not cv_config.enforcementPolicyConfig.enabled:
        cv_config.enforcementPolicyConfig.enabled = False
    return cv_config