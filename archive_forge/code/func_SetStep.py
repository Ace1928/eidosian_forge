from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def SetStep(self, step):
    """Set the step for this cluster.

    Can only be called on leaf nodes that have not yet had their step set.

    Args:
      step: The step that this cluster represents.
    """
    assert not self.__children
    assert not self.__step
    self.__step = step