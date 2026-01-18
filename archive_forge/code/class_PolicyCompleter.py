from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
class PolicyCompleter(completers.ListCommandCompleter):

    def __init__(self, api_version, **kwargs):
        super(PolicyCompleter, self).__init__(collection='dns.policies', api_version=api_version, list_command='alpha dns policies list --format=value(name)', parse_output=True, **kwargs)