from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.containeranalysis import util as containeranalysis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.images.packages import exceptions
from googlecloudsdk.command_lib.compute.images.packages import filter_utils
from googlecloudsdk.command_lib.compute.images.packages import flags as package_flags
def _GetVersions(self, image_packages, image_name):
    package_versions = {}
    for occurrence in image_packages:
        package_name = occurrence.installation.name
        versions = []
        for location in occurrence.installation.location:
            versions.append((location.version.name, location.version.revision))
        package_versions[package_name] = versions
    if not package_versions:
        raise exceptions.ImagePackagesInfoUnavailableException(image_name)
    return package_versions