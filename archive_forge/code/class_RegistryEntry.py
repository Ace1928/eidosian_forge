from (runtime, environment) to arbitrary data. Its main feature is that it
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from six.moves import map  # pylint:disable=redefined-builtin
class RegistryEntry(object):
    """An entry in the Registry.

  Attributes:
    runtime: str or re.RegexObject, the runtime to be staged
    envs: set(env.Environment), the environments to be staged
  """

    def __init__(self, runtime, envs):
        self.runtime = runtime
        self.envs = envs

    def _RuntimeMatches(self, runtime):
        try:
            return self.runtime.match(runtime)
        except AttributeError:
            return self.runtime == runtime

    def _EnvMatches(self, env):
        return env in self.envs

    def Matches(self, runtime, env):
        """Returns True iff the given runtime and environment match this entry.

    The runtime matches if it is an exact string match.

    The environment matches if it is an exact Enum match or if this entry has a
    "wildcard" (that is, None) for the environment.

    Args:
      runtime: str, the runtime to match
      env: env.Environment, the environment to match

    Returns:
      bool, whether the given runtime and environment match.
    """
        return self._RuntimeMatches(runtime) and self._EnvMatches(env)

    def __hash__(self):
        return hash((self.runtime, sum(sorted(map(hash, self.envs)))))

    def __eq__(self, other):
        return self.runtime == other.runtime and self.envs == other.envs

    def __ne__(self, other):
        return not self.__eq__(other)