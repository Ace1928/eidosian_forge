from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def AddUpgradeSoakingOverrideFlags(self, with_destructive=False):
    if with_destructive:
        group = self.parser.add_mutually_exclusive_group()
        self._AddRemoveUpgradeSoakingOverridesFlag(group)
        self._AddUpgradeSoakingOverrideFlags(group)
    else:
        self._AddUpgradeSoakingOverrideFlags(self.parser)