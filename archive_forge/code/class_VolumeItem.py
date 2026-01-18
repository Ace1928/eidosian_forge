from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.api_lib.kuberun import structuredout
class VolumeItem(structuredout.MapObject):

    @property
    def items(self):
        return [KeyToPath(x) for x in self._props.get('items', [])]