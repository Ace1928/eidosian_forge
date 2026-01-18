import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
class HasPermissions(Matcher):
    """Matches if a file has the given permissions.

    Permissions are specified and matched as a four-digit octal string.
    """

    def __init__(self, octal_permissions):
        """Construct a HasPermissions matcher.

        :param octal_permissions: A four digit octal string, representing the
            intended access permissions. e.g. '0775' for rwxrwxr-x.
        """
        super().__init__()
        self.octal_permissions = octal_permissions

    def match(self, filename):
        permissions = oct(os.stat(filename).st_mode)[-4:]
        return Equals(self.octal_permissions).match(permissions)