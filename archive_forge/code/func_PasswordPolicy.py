from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def PasswordPolicy(sql_messages, password_policy_min_length=None, password_policy_complexity=None, password_policy_reuse_interval=None, password_policy_disallow_username_substring=None, password_policy_password_change_interval=None, enable_password_policy=None, clear_password_policy=None):
    """Generates or clears password policy for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    password_policy_min_length: int, Minimum number of characters allowed.
    password_policy_complexity: string, The complexity of the password.
    password_policy_reuse_interval: int, Number of previous passwords that
      cannot be reused.
    password_policy_disallow_username_substring: boolean, True if disallow
      username as a part of the password.
    password_policy_password_change_interval: duration, Minimum interval at
      which password can be changed.
    enable_password_policy: boolean, True if password validation policy is
      enabled.
    clear_password_policy: boolean, True if clear existing password policy.

  Returns:
    sql_messages.PasswordValidationPolicy or None
  """
    should_generate_policy = any([password_policy_min_length is not None, password_policy_complexity is not None, password_policy_reuse_interval is not None, password_policy_disallow_username_substring is not None, password_policy_password_change_interval is not None, enable_password_policy is not None])
    if not should_generate_policy or clear_password_policy:
        return None
    password_policy = sql_messages.PasswordValidationPolicy()
    if password_policy_min_length is not None:
        password_policy.minLength = password_policy_min_length
    if password_policy_complexity is not None:
        password_policy.complexity = _ParseComplexity(sql_messages, password_policy_complexity)
    if password_policy_reuse_interval is not None:
        password_policy.reuseInterval = password_policy_reuse_interval
    if password_policy_disallow_username_substring is not None:
        password_policy.disallowUsernameSubstring = password_policy_disallow_username_substring
    if password_policy_password_change_interval is not None:
        password_policy.passwordChangeInterval = str(password_policy_password_change_interval) + 's'
    if enable_password_policy is not None:
        password_policy.enablePasswordPolicy = enable_password_policy
    return password_policy