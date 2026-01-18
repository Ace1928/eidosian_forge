import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def _set_full_version(self, version):
    m = self.re_valid_version.match(version)
    if not m:
        raise ValueError('Invalid version string %r' % version)
    if m.group('epoch') is None and ':' in m.group('upstream_version'):
        raise ValueError('Invalid version string %r' % version)
    self.__full_version = version
    self.__epoch = m.group('epoch')
    self.__upstream_version = m.group('upstream_version')
    self.__debian_revision = m.group('debian_revision')