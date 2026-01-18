from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util import completers
class OrganizationCompleter(completers.ResourceParamCompleter):
    """The organization completer."""

    def __init__(self, **kwargs):
        super(OrganizationCompleter, self).__init__(collection='cloudresourcemanager.organizations', list_command='organizations list --uri', param='organizationsId', **kwargs)