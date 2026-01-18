from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def ValidateDimensionAndValue(self, dim_name, dim_value):
    """Validates that a matrix dimension has a valid name and value."""
    if dim_name == _MODEL_DIMENSION:
        if dim_value not in self._model_ids:
            raise exceptions.ModelNotFoundError(dim_value)
    elif dim_name == _VERSION_DIMENSION:
        if dim_value not in self._version_ids:
            raise exceptions.VersionNotFoundError(dim_value)
    elif dim_name == _LOCALE_DIMENSION:
        if dim_value not in self._locale_ids:
            raise exceptions.LocaleNotFoundError(dim_value)
    elif dim_name == _ORIENTATION_DIMENSION:
        if dim_value not in self._orientation_ids:
            raise exceptions.OrientationNotFoundError(dim_value)
    else:
        raise exceptions.InvalidDimensionNameError(dim_name)
    return dim_value