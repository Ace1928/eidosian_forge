from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
def GetTempZipFileName(storage_url):
    """Returns temporary name for a temporarily compressed file."""
    return '%s_.gztmp' % storage_url.object_name