from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def DeprecationAction(flag_name, show_message=lambda _: True, show_add_help=lambda _: True, warn='Flag {flag_name} is deprecated.', error='Flag {flag_name} has been removed.', removed=False, action=None):
    """Prints a warning or error message for a flag that is being deprecated.

  Uses a _PreActionHook to wrap any existing Action on the flag and
  also adds deprecation messaging to flag help.

  Args:
    flag_name: string, name of flag to be deprecated
    show_message: callable, boolean function that takes the argument value
        as input, validates it against some criteria and returns a boolean.
        If true deprecation message is shown at runtime. Deprecation message
        will always be appended to flag help.
    show_add_help: boolean, whether to show additional help in help text.
    warn: string, warning message, 'flag_name' template will be replaced with
        value of flag_name parameter
    error: string, error message, 'flag_name' template will be replaced with
        value of flag_name parameter
    removed: boolean, if True warning message will be printed when show_message
        fails, if False error message will be printed
    action: argparse.Action, action to be wrapped by this action

  Returns:
    argparse.Action, deprecation action to use.
  """
    if removed:
        add_help = _AdditionalHelp('(REMOVED)', error.format(flag_name=flag_name))
    else:
        add_help = _AdditionalHelp('(DEPRECATED)', warn.format(flag_name=flag_name))
    if not action:
        action = 'store'

    def DeprecationFunc(value):
        if show_message(value):
            if removed:
                raise parser_errors.ArgumentError(add_help.message)
            else:
                log.warning(add_help.message)
    if show_add_help:
        return _PreActionHook(action, DeprecationFunc, add_help)
    return _PreActionHook(action, DeprecationFunc, None)