import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def _safe_get_entry_point_description(self, ep, group):
    ep.dist.activate()
    meta_group = 'paste.description.' + group
    meta = ep.dist.get_entry_info(meta_group, ep.name)
    if not meta:
        generic = list(pkg_resources.iter_entry_points(meta_group, 'generic'))
        if not generic:
            return super_generic(ep.load())
        obj = generic[0].load()
        desc = obj(ep, group)
    else:
        desc = meta.load()
    return desc