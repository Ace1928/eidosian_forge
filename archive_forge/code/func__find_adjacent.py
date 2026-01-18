from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping, MutableSet, MutableSequence
from pathlib import Path
from ansible import constants as C
from ansible.release import __version__ as ansible_version
from ansible.errors import AnsibleError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.plugin_docs import read_docstring
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def _find_adjacent(path, plugin, extensions):
    adjacent = Path(path)
    plugin_base_name = plugin.split('.')[-1]
    if adjacent.stem != plugin_base_name:
        adjacent = adjacent.with_name(plugin_base_name)
    paths = []
    for ext in extensions:
        candidate = adjacent.with_suffix(ext)
        if candidate == adjacent:
            continue
        if candidate.exists():
            paths.append(to_native(candidate))
    return paths