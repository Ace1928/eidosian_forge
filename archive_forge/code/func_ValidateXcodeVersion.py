from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def ValidateXcodeVersion(self, xcode_version):
    """Validates that an Xcode version is in the TestEnvironmentCatalog."""
    if xcode_version not in [xv.version for xv in self.catalog.xcodeVersions]:
        raise exceptions.XcodeVersionNotFoundError(xcode_version)