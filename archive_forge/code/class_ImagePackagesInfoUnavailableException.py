from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
class ImagePackagesInfoUnavailableException(exceptions.ToolException):
    """Image info about packages is not available."""

    def __init__(self, image_name):
        error_message = "Package information for '{}' not available:\n - Please refer to product documentation for additional information.\n ".format(image_name)
        super(ImagePackagesInfoUnavailableException, self).__init__(error_message)