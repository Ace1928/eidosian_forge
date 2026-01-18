from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PatchAndConfigFeatureSetValueValuesEnum(_messages.Enum):
    """Currently set PatchAndConfigFeatureSet for name.

    Values:
      PATCH_AND_CONFIG_FEATURE_SET_UNSPECIFIED: Not specified placeholder
      OSCONFIG_B: Basic feature set. Enables only the basic set of features.
      OSCONFIG_C: Classic set of functionality.
    """
    PATCH_AND_CONFIG_FEATURE_SET_UNSPECIFIED = 0
    OSCONFIG_B = 1
    OSCONFIG_C = 2