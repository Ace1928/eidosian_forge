from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def _add_region_groups(self):
    region_transformed_group_names = self._setup_nested_groups('region', self.regions_lookup, self.regions_parent_lookup)
    for site_id in self.sites_lookup:
        region_id = self.sites_region_lookup.get(site_id, None)
        if region_id is None:
            continue
        self.inventory.add_child(region_transformed_group_names[region_id], self.site_group_names[site_id])