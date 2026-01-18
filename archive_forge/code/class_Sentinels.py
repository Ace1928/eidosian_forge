from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core.credentials import creds as c_creds
class Sentinels(object):
    """Holder for sentinel file locations.

  Attributes:
    config_sentinel: str, The path to the sentinel that indicates changes were
      made to properties or the active configuration.
  """

    def __init__(self, config_sentinel):
        self.config_sentinel = config_sentinel