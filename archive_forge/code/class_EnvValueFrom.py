from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.api_lib.kuberun import structuredout
class EnvValueFrom(structuredout.MapObject):
    """Represents the ValueFrom field of an EnvVar."""

    @property
    def secretKeyRef(self):
        if self._props.get('secretKeyRef'):
            return SecretKey(self._props.get('secretKeyRef'))
        else:
            return None

    @property
    def configMapKeyRef(self):
        if self._props.get('configMapKeyRef'):
            return ConfigMapKey(self._props.get('configMapKeyRef'))
        else:
            return None