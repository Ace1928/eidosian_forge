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
def _add_site_groups(self):
    self.site_group_names = dict()
    for site_id, site_name in self.sites_lookup_slug.items():
        site_group_name = self.generate_group_name(self._pluralize_group_by('site'), site_name)
        site_transformed_group_name = self.inventory.add_group(group=site_group_name)
        self.site_group_names[site_id] = site_transformed_group_name