from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import lexer
import six
def SetContext(self, context=None):
    """Sets the default command prompt context."""
    self.context = six.text_type(context or '')