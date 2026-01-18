from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def GetOrAddChild(self, piece_name):
    """Return the cluster representing the given piece_name within this cluster.

    Args:
      piece_name: String representing the piece name of the desired child.
    Returns:
      Cluster representing the child.
    """
    assert not self.__step
    if piece_name not in self.__children:
        self.__children[piece_name] = _Cluster(self, piece_name)
    return self.__children[piece_name]