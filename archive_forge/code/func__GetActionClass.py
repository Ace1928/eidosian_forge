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
def _GetActionClass(self):
    if isinstance(self.wrapped_action, six.string_types):
        action_cls = GetArgparseBuiltInAction(self.wrapped_action)
    else:
        action_cls = self.wrapped_action
    return action_cls