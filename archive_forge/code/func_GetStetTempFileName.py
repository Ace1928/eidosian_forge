from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
def GetStetTempFileName(storage_url):
    """Returns temporary file name for result of STET transform."""
    return '%s_.stet_tmp' % storage_url.object_name