from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class Exec(Type):
    """Coerces the value to accommodate Docker CMD/ENTRYPOINT requirements.

  Validates the value is a string, then tries to modify the string (if
  necessary) so that the command represented will become PID 1 inside the
  Docker container. See Docker documentation on "docker kill" for more info:
  https://docs.docker.com/engine/reference/commandline/kill/

  If the command already starts with `exec` or appears to be in "exec form"
  (starts with `[`), no further action is needed. Otherwise, prepend the
  command with `exec` so that it will become PID 1 on execution.
  """

    def __init__(self, default=None):
        """Initialize parent, a converting type validator for `str`."""
        super(Exec, self).__init__(str, convert=True, default=default)

    def Validate(self, value, key):
        """Validate according to parent behavior and coerce to start with `exec`."""
        value = super(Exec, self).Validate(value, key)
        if value.startswith('[') or value.startswith('exec'):
            return value
        else:
            return 'exec ' + value