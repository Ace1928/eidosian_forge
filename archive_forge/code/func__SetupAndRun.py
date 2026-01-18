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
@staticmethod
def _SetupAndRun(group, check_list):
    """Builds up linter and executes it for given set of checks."""
    linter = Linter()
    unknown_checks = []
    if not check_list:
        for check in _DEFAULT_LINT_CHECKS:
            linter.AddCheck(check)
    else:
        available_checkers = dict(((checker.name, checker) for checker in _DEFAULT_LINT_CHECKS + _LINT_CHECKS))
        for check in check_list:
            if check in available_checkers:
                linter.AddCheck(available_checkers[check])
            else:
                unknown_checks.append(check)
    if unknown_checks:
        raise UnknownCheckException('Unknown lint checks: %s' % ','.join(unknown_checks))
    return linter.Run(group)