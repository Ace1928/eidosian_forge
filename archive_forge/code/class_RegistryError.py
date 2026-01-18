import os
import sys
import stat
import fnmatch
import collections
import errno
class RegistryError(Exception):
    """Raised when a registry operation with the archiving
    and unpacking registries fails"""