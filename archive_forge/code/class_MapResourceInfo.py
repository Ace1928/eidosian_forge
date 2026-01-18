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
class MapResourceInfo(ResourceInfo):
    """Store the mapping of one resource type to another.

    like: OS::Networking::FloatingIp -> OS::Neutron::FloatingIp
    """
    description = 'Mapping'
    __slots__ = tuple()

    def get_class(self, files=None):
        return None

    def get_resource_info(self, resource_type=None, resource_name=None):
        return self.registry.get_resource_info(self.value, resource_name)