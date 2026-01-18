from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
@classmethod
def New(cls, client, namespace):
    """Produces a new Task object.

    Args:
      client: The Cloud Run API client.
      namespace: str, The serving namespace.

    Returns:
      A new Task object.
    """
    ret = super(Task, cls).New(client, namespace)
    ret.spec.template.spec.containers = [client.MESSAGES_MODULE.Container()]
    return ret