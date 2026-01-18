from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.events import iam_util
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _fetch_product_type(client):
    """Fetch eventing product type inferred by namespaces."""
    namespaces_list = client.ListNamespaces()
    if events_constants.KUBERUN_EVENTS_NAMESPACE in namespaces_list:
        return events_constants.Product.KUBERUN
    elif events_constants.CLOUDRUN_EVENTS_NAMESPACE in namespaces_list:
        return events_constants.Product.CLOUDRUN
    else:
        raise exceptions.EventingInstallError('Neither CloudRun nor KubeRun events installed')