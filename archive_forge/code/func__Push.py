from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def _Push(self, token, value):
    """Push values to stack at start of nesting.

    When a new object scope is beginning, will push the token (type of scope)
    along with the new objects value, the latter of which is provided through
    the various build methods of the builder.

    Args:
      token: Token indicating the type of scope which is being created; must
        belong to _TOKEN_VALUES.
      value: Value to associate with given token.  Construction of value is
        determined by the builder provided to this handler at construction.
    """
    self._top = (token, value)
    self._stack.append(self._top)