from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
def _ensure_location_field(req, messages):
    """Ensures that the location field is set."""
    if not req.snapshotSettings:
        req.snapshotSettings = messages.SnapshotSettings()
    if not req.snapshotSettings.storageLocation:
        req.snapshotSettings.storageLocation = messages.SnapshotSettingsStorageLocationSettings()