from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core.console import console_io
def ParsePolicyFile(policy_file_path, policy_message_type):
    """Constructs an IAM Policy message from a JSON/YAML formatted file.

  Args:
    policy_file_path: Path to the JSON or YAML IAM policy file.
    policy_message_type: Policy message type to convert JSON or YAML to.
  Returns:
    a protorpc.Message of type policy_message_type filled in from the JSON or
    YAML policy file.
  Raises:
    BadFileException if the JSON or YAML file is malformed.
  """
    policy, unused_mask = iam_util.ParseYamlOrJsonPolicyFile(policy_file_path, policy_message_type)
    if not policy.etag:
        msg = 'The specified policy does not contain an "etag" field identifying a specific version to replace. Changing a policy without an "etag" can overwrite concurrent policy changes.'
        console_io.PromptContinue(message=msg, prompt_string='Replace existing policy', cancel_on_no=True)
    return policy