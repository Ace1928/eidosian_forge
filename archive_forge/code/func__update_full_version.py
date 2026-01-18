import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def _update_full_version(self):
    version = ''
    if self.__epoch is not None:
        version += self.__epoch + ':'
    version += self.__upstream_version
    if self.__debian_revision:
        version += '-' + self.__debian_revision
    self.full_version = version