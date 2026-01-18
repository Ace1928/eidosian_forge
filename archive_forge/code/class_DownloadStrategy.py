from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class DownloadStrategy(object):
    """Enum class for specifying download strategy."""
    ONE_SHOT = 'oneshot'
    RESUMABLE = 'resumable'