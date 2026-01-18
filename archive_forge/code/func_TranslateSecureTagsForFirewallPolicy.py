from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import tag_utils
def TranslateSecureTagsForFirewallPolicy(client, secure_tags):
    """Returns a list of firewall policy rule secure tags, translating namespaced tags if needed.

  Args:
    client: compute client
    secure_tags: array of secure tag values

  Returns:
    List of firewall policy rule secure tags
  """
    ret_secure_tags = []
    for tag in secure_tags:
        name = TranslateSecureTag(tag)
        ret_secure_tags.append(client.messages.FirewallPolicyRuleSecureTag(name=name))
    return ret_secure_tags