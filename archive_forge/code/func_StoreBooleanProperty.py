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
def StoreBooleanProperty(prop):
    """Get an argparse action that stores a value in a Boolean property.

  Handles auto-generated --no-* inverted flags by inverting the value.

  Also stores the value in the namespace object, like the default action. The
  value is stored in the invocation stack, rather than persisted permanently.

  Args:
    prop: properties._Property, The property that should get the invocation
        value.

  Returns:
    argparse.Action, An argparse action that routes the value correctly.
  """

    class Action(argparse.Action):
        """The action created for StoreBooleanProperty."""
        store_property = (prop, 'bool', None)

        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            if 'nargs' not in kwargs:
                kwargs['nargs'] = 0
            option_strings = kwargs.get('option_strings')
            if option_strings:
                option_string = option_strings[0]
            else:
                option_string = None
            if option_string and option_string.startswith('--no-'):
                self._inverted = True
                kwargs['nargs'] = 0
                kwargs['const'] = None
                kwargs['choices'] = None
            else:
                self._inverted = False
            super(Action, self).__init__(*args, **kwargs)
            properties.VALUES.SetInvocationValue(prop, None, option_string)
            if '_ARGCOMPLETE' in os.environ:
                self._orig_class = argparse._StoreAction

        def __call__(self, parser, namespace, values, option_string=None):
            if self._inverted:
                if values in ('true', []):
                    values = 'false'
                else:
                    values = 'false'
            elif values == []:
                values = 'true'
            properties.VALUES.SetInvocationValue(prop, values, option_string)
            setattr(namespace, self.dest, values)
    return Action