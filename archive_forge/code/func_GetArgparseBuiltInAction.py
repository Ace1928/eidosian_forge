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
def GetArgparseBuiltInAction(action):
    """Get an argparse.Action from a string.

  This function takes one of the supplied argparse.Action strings (see below)
  and returns the corresponding argparse.Action class.

  This "work around" is (e.g. hack) is necessary due to the fact these required
  action mappings are only exposed through subclasses of
  argparse._ActionsContainer as opposed to a static function or global variable.

  Args:
    action: string, one of the following supplied argparse.Action names:
      'store', 'store_const', 'store_false', 'append', 'append_const', 'count',
      'version', 'parsers'.

  Returns:
    argparse.Action, the action class to use.

  Raises:
    ValueError: For unknown action string.
  """
    fake_actions_container = argparse._ActionsContainer(description=None, prefix_chars=None, argument_default=None, conflict_handler='error')
    action_cls = fake_actions_container._registry_get('action', action)
    if action_cls is None:
        raise ValueError('unknown action "{0}"'.format(action))
    return action_cls