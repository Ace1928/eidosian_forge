from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class HybridNonAlphaConfigError(exceptions.Error):
    """Hybrid Configs are currently only supported in the alpha release track."""

    def __init__(self):
        msg = 'invalid config file.'
        super(HybridNonAlphaConfigError, self).__init__(msg)