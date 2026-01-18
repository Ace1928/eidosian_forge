from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class VolumesAsDictionaryWrapper(k8s_object.ListAsDictionaryWrapper):
    """Wraps a list of volumes in a dict-like object.

  Additionally provides properties to access volumes of specific type in a
  mutable dict-like object.
  """

    def __init__(self, volumes_to_wrap, volume_class):
        """Wraps a list of volumes in a dict-like object.

    Args:
      volumes_to_wrap: list[Volume], list of volumes to treat as a dict.
      volume_class: type of the underlying Volume objects.
    """
        super(VolumesAsDictionaryWrapper, self).__init__(volumes_to_wrap)
        self._volumes = volumes_to_wrap
        self._volume_class = volume_class

    @property
    def secrets(self):
        """Mutable dict-like object for volumes with a secret source type."""
        return k8s_object.KeyValueListAsDictionaryWrapper(self._volumes, self._volume_class, value_field='secret', filter_func=lambda volume: volume.secret is not None)

    @property
    def config_maps(self):
        """Mutable dict-like object for volumes with a config map source type."""
        return k8s_object.KeyValueListAsDictionaryWrapper(self._volumes, self._volume_class, value_field='configMap', filter_func=lambda volume: volume.configMap is not None)