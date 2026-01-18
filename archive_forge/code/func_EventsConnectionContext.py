from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.run import connection_context as run_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.core import exceptions
def EventsConnectionContext(args):
    """Returns the appropriate cluster connection context based on args.

  Unless the user has configured cluster connectivity options, calling this
  will result in the user being prompted to select a GKE cluster.

  Args:
    args: A parsed argument context

  Returns:
    googlecloudsdk.command_lib.run.connection_context.ConnectionInfo

  Raises:
    flags.ConfigurationError when the user has not specified a cluster
    connection method and can't be prompted.
  """
    api_name = _CLUSTER_EVENTS_API_NAME
    api_version = _CLUSTER_EVENTS_API_VERSION
    connection = flags.ClusterConnectionMethod(args)
    if connection == flags.CONNECTION_KUBECONFIG:
        kubeconfig_path, context = flags.KubeconfigPathAndContext(args)
        kubeconfig = run_flags.GetKubeconfig(kubeconfig_path)
        return run_context.KubeconfigConnectionContext(kubeconfig, api_name, api_version, context)
    elif connection == flags.CONNECTION_GKE:
        cluster_ref = flags.ParseClusterRefOrPromptUser(args)
        return run_context.GKEConnectionContext(cluster_ref, api_name, api_version)
    else:
        raise exceptions.Error('Unable to determine cluster connection method')