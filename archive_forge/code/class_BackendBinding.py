from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import kubernetesobject
class BackendBinding(kubernetesobject.KubernetesObject):
    """Wraps JSON-based dict object of a backend binding."""

    @property
    def service(self):
        return self._props['spec']['targetService']