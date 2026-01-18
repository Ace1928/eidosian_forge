from (runtime, environment) to arbitrary data. Its main feature is that it
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from six.moves import map  # pylint:disable=redefined-builtin
def _RuntimeMatches(self, runtime):
    try:
        return self.runtime.match(runtime)
    except AttributeError:
        return self.runtime == runtime