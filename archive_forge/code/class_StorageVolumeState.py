from libcloud.common.types import (
class StorageVolumeState(Type):
    """
    Standard states of a StorageVolume
    """
    AVAILABLE = 'available'
    ERROR = 'error'
    INUSE = 'inuse'
    CREATING = 'creating'
    DELETING = 'deleting'
    DELETED = 'deleted'
    BACKUP = 'backup'
    ATTACHING = 'attaching'
    UNKNOWN = 'unknown'
    MIGRATING = 'migrating'
    UPDATING = 'updating'