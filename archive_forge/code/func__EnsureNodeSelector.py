from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
def _EnsureNodeSelector(self):
    if self.spec.nodeSelector is None:
        self.spec.nodeSelector = k8s_object.InitializedInstance(self._messages.RevisionSpec.NodeSelectorValue)