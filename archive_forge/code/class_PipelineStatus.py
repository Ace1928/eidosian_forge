from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class PipelineStatus(GitlabEnum):
    CREATED: str = 'created'
    WAITING_FOR_RESOURCE: str = 'waiting_for_resource'
    PREPARING: str = 'preparing'
    PENDING: str = 'pending'
    RUNNING: str = 'running'
    SUCCESS: str = 'success'
    FAILED: str = 'failed'
    CANCELED: str = 'canceled'
    SKIPPED: str = 'skipped'
    MANUAL: str = 'manual'
    SCHEDULED: str = 'scheduled'