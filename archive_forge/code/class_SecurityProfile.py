from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityProfile(_messages.Message):
    """User selected security profile

  Fields:
    disableRuntimeRules: Don't apply runtime rules. When set to true, no
      objects/deployments will be installed in the cluster to enforce runtime
      rules. This is useful to work with config-as-code systems
    name: Name with version of selected security profile A security profile
      name follows kebob-case (a-zA-Z*) and a version is like MAJOR.MINOR-
      suffix suffix is ([a-zA-Z0-9\\-_\\.]+) e.g. default-1.0-gke.0
  """
    disableRuntimeRules = _messages.BooleanField(1)
    name = _messages.StringField(2)