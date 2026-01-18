from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def KubeconfigPathAndContext(args):
    """Returns a 2-tuple of (kubeconfig, context).

  The kubeconfig path and context returned will be those specified in args
  or those coming from properties. Missing values for kubeconfig or context
  will be returned as None values.

  Args:
    args: Parsed argument context object

  Returns:
    2-tuple of (kubeconfig, context) where the kubeconfig is the path to the
    a kubeconfig file and the context is the name of the context to be used.
  """
    kubeconfig = args.kubeconfig if args.IsSpecified('kubeconfig') else properties.VALUES.kuberun.kubeconfig.Get()
    context = args.context if args.IsSpecified('context') else properties.VALUES.kuberun.context.Get()
    return (kubeconfig, context)