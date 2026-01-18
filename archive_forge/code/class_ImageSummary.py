from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ImageSummary(object):
    """ImageSummary is a container class whose structure creates command output.
  """

    def __init__(self, registry, repository, digest):
        self.fully_qualified_digest = '{registry}/{repository}@{digest}'.format(registry=registry, repository=repository, digest=digest)
        self.registry = registry
        self.repository = repository
        self.digest = digest