import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
Construct a HasPermissions matcher.

        :param octal_permissions: A four digit octal string, representing the
            intended access permissions. e.g. '0775' for rwxrwxr-x.
        