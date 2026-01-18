from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def _CheckMapShorthand(self):
    """Checks for N '*' chars shorthand for .map(N)."""
    map_level = 0
    while self.IsCharacter('*'):
        map_level += 1
    if not map_level:
        return
    self._expr = '{}map({}).{}'.format(self._expr[:self._position - map_level], map_level, self._expr[self._position:])
    self._position -= map_level