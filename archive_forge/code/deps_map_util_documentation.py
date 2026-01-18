import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
Validates fallthrough map to ensure fallthrough map is not invalid.

  Fallthrough maps are only invalid if an inactive fallthrough comes before
  an active fallthrough. It could result in an active fallthrough that can
  never be reached.

  Args:
    fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs we are validating

  Returns:
    (bool, str), bool for whether fallthrough map is valid and str for
      the error message
  