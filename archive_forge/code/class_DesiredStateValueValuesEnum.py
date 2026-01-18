from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DesiredStateValueValuesEnum(_messages.Enum):
    """Default is INSTALLED. The desired state the agent should maintain for
    this recipe. INSTALLED: The software recipe is installed on the instance
    but won't be updated to new versions. UPDATED: The software recipe is
    installed on the instance. The recipe is updated to a higher version, if a
    higher version of the recipe is assigned to this instance. REMOVE: Remove
    is unsupported for software recipes and attempts to create or update a
    recipe to the REMOVE state is rejected.

    Values:
      DESIRED_STATE_UNSPECIFIED: The default is to ensure the package is
        installed.
      INSTALLED: The agent ensures that the package is installed.
      UPDATED: The agent ensures that the package is installed and
        periodically checks for and install any updates.
      REMOVED: The agent ensures that the package is not installed and
        uninstall it if detected.
    """
    DESIRED_STATE_UNSPECIFIED = 0
    INSTALLED = 1
    UPDATED = 2
    REMOVED = 3