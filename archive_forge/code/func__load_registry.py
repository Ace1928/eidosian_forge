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
def _load_registry(self, path, registry):
    for k, v in iter(registry.items()):
        if v is None:
            self._register_info(path + [k], None)
        elif is_hook_definition(k, v) or is_valid_restricted_action(k, v):
            self._register_item(path + [k], v)
        elif isinstance(v, dict):
            self._load_registry(path + [k], v)
        else:
            self._register_info(path + [k], ResourceInfo(self, path + [k], v))