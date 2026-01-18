from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
def _EnsureResources(self):
    limits_cls = self._messages.ResourceRequirements.LimitsValue
    if self.resources is not None:
        if self.resources.limits is None:
            self.resources.limits = k8s_object.InitializedInstance(limits_cls)
    else:
        self.resources = k8s_object.InitializedInstance(self._messages.ResourceRequirements)