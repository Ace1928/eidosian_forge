from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
class CyclicDependencyError(Exception):

    def __init__(self, leftover_dependency_map):
        """Creates a CyclicDependencyException."""
        self.leftover_dependency_map = leftover_dependency_map
        super(CyclicDependencyError, self).__init__()