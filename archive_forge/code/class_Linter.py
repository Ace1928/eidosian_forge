from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
class Linter(object):
    """Lints gcloud commands."""

    def __init__(self):
        self._checks = []

    def AddCheck(self, check):
        self._checks.append(check())

    def Run(self, group_root):
        """Runs registered checks on all groups and commands."""
        for group in _WalkGroupTree(group_root):
            for check in self._checks:
                check.ForEveryGroup(group)
            for command in six.itervalues(group.commands):
                for check in self._checks:
                    check.ForEveryCommand(command)
        return [issue for check in self._checks for issue in check.End()]