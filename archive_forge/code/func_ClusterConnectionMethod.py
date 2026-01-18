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
def ClusterConnectionMethod(args):
    """Returns the connection method resulting from args and configuration.

  This functionality is broken out so that it can be used as a means to
  determine whether the user should be prompted to select a cluster, although
  the user is not prompted as part of this function's execution.

  Args:
    args: Namespace of parsed args

  Returns:
     Constant, one of CONNECTION_GKE or CONNECTION_KUBECONFIG.

  Raises:
    ConfigurationError: when the configuration is invalid.
  """
    if args.IsSpecified('cluster') or args.IsSpecified('cluster_location'):
        return CONNECTION_GKE
    if args.IsSpecified('use_kubeconfig') or args.IsSpecified('kubeconfig') or args.IsSpecified('context'):
        return CONNECTION_KUBECONFIG
    configured_kubeconfig_options = _ExplicitlySetProperties(['kubeconfig', 'use_kubeconfig', 'context'])
    configured_gke_options = _ExplicitlySetProperties(['cluster', 'cluster_location'])
    if configured_kubeconfig_options and configured_gke_options:
        raise ConfigurationError('Multiple cluster connection options are configured. To remove one of the options, run `{}` or `{}`.'.format(_UnsetCommandsAsString(configured_kubeconfig_options), _UnsetCommandsAsString(configured_gke_options)))
    if configured_kubeconfig_options:
        return CONNECTION_KUBECONFIG
    return CONNECTION_GKE