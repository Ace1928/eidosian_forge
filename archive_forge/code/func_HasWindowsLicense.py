from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def HasWindowsLicense(resource, resource_parser):
    """Returns True if the given image or disk has a Windows license."""
    for license_uri in resource.licenses:
        license_ref = resource_parser.Parse(license_uri, collection='compute.licenses')
        if license_ref.project in constants.WINDOWS_IMAGE_PROJECTS:
            return True
    return False