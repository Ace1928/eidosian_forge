from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetKubeconfigPathFlag():
    """Anthos auth token kubeconfig-path flag, specifies the Path to the kubeconfig path that would be updated with ID and access token on expiry."""
    return base.Argument('--kubeconfig-path', required=False, help='Path to the kubeconfig path that would be updated with ID and access token on expiry.')