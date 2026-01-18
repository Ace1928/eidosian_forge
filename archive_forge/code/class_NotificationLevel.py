from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class NotificationLevel(GitlabEnum):
    DISABLED: str = 'disabled'
    PARTICIPATING: str = 'participating'
    WATCH: str = 'watch'
    GLOBAL: str = 'global'
    MENTION: str = 'mention'
    CUSTOM: str = 'custom'