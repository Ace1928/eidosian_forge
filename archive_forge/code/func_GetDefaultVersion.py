from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def GetDefaultVersion(self):
    """Return the default version listed in the iOS environment catalog."""
    version = self._default_version if self._default_version else self._FindDefaultDimension(self.catalog.versions)
    if not version:
        raise exceptions.DefaultDimensionNotFoundError(_VERSION_DIMENSION)
    return version