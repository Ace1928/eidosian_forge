from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class AccessLevel(IntEnum):
    NO_ACCESS: int = 0
    MINIMAL_ACCESS: int = 5
    GUEST: int = 10
    REPORTER: int = 20
    DEVELOPER: int = 30
    MAINTAINER: int = 40
    OWNER: int = 50
    ADMIN: int = 60