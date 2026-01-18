from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ContainerData(object):
    """ContainerData objects get returned from a command for formatted output.
  """

    def __init__(self, registry, repository, digest):
        self.image_summary = ImageSummary(registry, repository, digest)