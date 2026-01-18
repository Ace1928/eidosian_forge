from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
def _AngrySnakeCase(self, key):
    """Returns an ANGRY_SNAKE_CASE string representation of a parsed key.

    For key input [A, B, C] the headings [C, C_B, C_B_A] are generated. The
    first heading not in self._snake_headings is added to self._snake_headings
    and returned.

    Args:
        key: A parsed resource key and/or list of strings.

    Returns:
      The ANGRY_SNAKE_CASE string representation of key, adding components
        from right to left to disambiguate from previous ANGRY_SNAKE_CASE
        strings.
    """
    if self._snake_re is None:
        self._snake_re = re.compile('((?<=[a-z0-9])[A-Z]+|(?!^)[A-Z](?=[a-z]))')
    label = ''
    for index in reversed(key):
        if isinstance(index, six.string_types):
            key_snake = self._snake_re.sub('_\\1', index).upper()
            if label:
                label = key_snake + '_' + label
            else:
                label = key_snake
            if label not in self._snake_headings:
                self._snake_headings[label] = 1
                break
    return label