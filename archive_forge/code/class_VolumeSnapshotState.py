from libcloud.common.types import (
class VolumeSnapshotState(Type):
    """
    Standard states of VolumeSnapshots
    """
    AVAILABLE = 'available'
    ERROR = 'error'
    CREATING = 'creating'
    DELETING = 'deleting'
    RESTORING = 'restoring'
    UNKNOWN = 'unknown'
    UPDATING = 'updating'