import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SetBaseConfiguration(self, value):
    """Sets the build configuration in all child XCBuildConfiguration objects.
    """
    for configuration in self._properties['buildConfigurations']:
        configuration.SetBaseConfiguration(value)