from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateSubscriptionResourceArg(verb, plural=False):
    """Create a resource argument for a Cloud Pub/Sub Subscription.

  Args:
    verb: str, the verb to describe the resource, such as 'to update'.
    plural: bool, if True, use a resource argument that returns a list.

  Returns:
    the PresentationSpec for the resource argument.
  """
    if plural:
        help_stem = 'One or more subscriptions'
    else:
        help_stem = 'Name of the subscription'
    return presentation_specs.ResourcePresentationSpec('subscription', GetSubscriptionResourceSpec(), '{} {}'.format(help_stem, verb), required=True, plural=plural, prefixes=True)