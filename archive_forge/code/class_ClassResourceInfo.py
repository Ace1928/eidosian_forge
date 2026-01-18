import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
class ClassResourceInfo(ResourceInfo):
    """Store the mapping of resource name to python class implementation."""
    description = 'Plugin'
    __slots__ = tuple()

    def get_class(self, files=None):
        return self.value