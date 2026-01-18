from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class FileOutputError(exceptions.Error):
    """Error thrown for issues with writing to files."""