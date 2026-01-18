from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def GetDefaultModel(self):
    """Return the default model listed in the iOS environment catalog."""
    model = self._default_model if self._default_model else self._FindDefaultDimension(self.catalog.models)
    if not model:
        raise exceptions.DefaultDimensionNotFoundError(_MODEL_DIMENSION)
    return model