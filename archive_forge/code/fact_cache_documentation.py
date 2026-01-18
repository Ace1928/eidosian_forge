from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.plugins.loader import cache_loader
from ansible.utils.display import Display
 Flush the fact cache of all keys. 