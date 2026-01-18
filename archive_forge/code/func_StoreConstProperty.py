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
def StoreConstProperty(prop, const):
    """Get an argparse action that stores a constant in a property.

  Also stores the constant in the namespace object, like the store_true action.
  The const is stored in the invocation stack, rather than persisted
  permanently.

  Args:
    prop: properties._Property, The property that should get the invocation
        value.
    const: str, The constant that should be stored in the property.

  Returns:
    argparse.Action, An argparse action that routes the value correctly.
  """

    class Action(argparse.Action):
        """The action created for StoreConstProperty."""
        store_property = (prop, 'value', const)

        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs['nargs'] = 0
            super(Action, self).__init__(*args, **kwargs)
            if '_ARGCOMPLETE' in os.environ:
                self._orig_class = argparse._StoreConstAction

        def __call__(self, parser, namespace, values, option_string=None):
            properties.VALUES.SetInvocationValue(prop, const, option_string)
            setattr(namespace, self.dest, const)
    return Action