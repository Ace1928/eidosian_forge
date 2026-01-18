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
class ClusterConnectionFlags(BinaryCommandFlag):
    """Encapsulates logic for handling flags used for connecting to a cluster."""

    def AddToParser(self, parser):
        from googlecloudsdk.command_lib.kuberun import resource_args
        mutex_group = parser.add_mutually_exclusive_group()
        concept_parsers.ConceptParser([resource_args.CLUSTER_PRESENTATION]).AddToParser(mutex_group)
        kubeconfig_group = mutex_group.add_group()
        kubeconfig_group.add_argument('--context', help='Name of the context in your kubectl config file to use for connecting. Cannot be specified together with --cluster and --cluster-location.')
        kubeconfig_group.add_argument('--kubeconfig', help='Absolute path to your kubectl config file. Cannot be specified together with --cluster and --cluster-location.')
        kubeconfig_group.add_argument('--use-kubeconfig', default=False, action='store_true', help='Use the kubectl config to connect to the cluster. If --kubeconfig is not also provided, the colon- or semicolon-delimited list of paths specified by $KUBECONFIG will be used. If $KUBECONFIG is unset, this defaults to ~/.kube/config. Cannot be specified together with --cluster and --cluster-location.')

    def FormatFlags(self, args):
        exec_args = []
        connection = ClusterConnectionMethod(args)
        if connection == CONNECTION_GKE:
            cluster_ref = ParseClusterRefOrPromptUser(args)
            exec_args.extend(['--cluster', cluster_ref.SelfLink()])
        elif connection == CONNECTION_KUBECONFIG:
            kubeconfig, context = KubeconfigPathAndContext(args)
            if kubeconfig:
                exec_args.extend(['--kubeconfig', kubeconfig])
            if context:
                exec_args.extend(['--context', context])
        return exec_args