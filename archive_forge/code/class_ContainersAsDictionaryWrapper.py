from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class ContainersAsDictionaryWrapper(k8s_object.ListAsDictionaryWrapper):
    """Wraps a list of containers in a mutable dict-like object mapping containers by name.

  Accessing a container name that does not exist will automatically add a new
  container with the specified name to the underlying list.
  """

    def __init__(self, containers_to_wrap, volumes, messages_mod):
        """Wraps a list of containers in a mutable dict-like object.

    Args:
      containers_to_wrap: list[Container], list of containers to treat as a
        dict.
      volumes: the volumes defined in the containing resource used to classify
        volume mounts
      messages_mod: the messages module
    """
        self._volumes = volumes
        self._messages = messages_mod
        super(ContainersAsDictionaryWrapper, self).__init__(ContainerSequenceWrapper(containers_to_wrap, volumes, messages_mod))

    def __getitem__(self, key):
        try:
            return super(ContainersAsDictionaryWrapper, self).__getitem__(key)
        except KeyError:
            container = Container(self._volumes, self._messages, name=key)
            self._m.append(container)
            return container

    def MakeSerializable(self):
        return super(ContainersAsDictionaryWrapper, self).MakeSerializable().MakeSerializable()