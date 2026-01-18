from __future__ import unicode_literals
import os.path  # splitext
import pkg_resources
from pybtex.exceptions import PybtexError
def _load_entry_point(group, name, use_aliases=False):
    groups = [group, group + '.aliases'] if use_aliases else [group]
    for search_group in groups:
        for entry_point in pkg_resources.iter_entry_points(search_group, name):
            return entry_point.load()
    raise PluginNotFound(group, name)