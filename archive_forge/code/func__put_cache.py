from __future__ import absolute_import, division, print_function
import json
import logging
import optparse
import os
import ssl
import sys
import time
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.module_utils.six import integer_types, text_type, string_types
from ansible.module_utils.six.moves import configparser
from psphere.client import Client
from psphere.errors import ObjectNotFoundError
from psphere.managedobjects import HostSystem, VirtualMachine, ManagedObject, ClusterComputeResource
from suds.sudsobject import Object as SudsObject
def _put_cache(self, name, value):
    """
        Saves the value to cache with the name given.
        """
    if self.config.has_option('defaults', 'cache_dir'):
        cache_dir = os.path.expanduser(self.config.get('defaults', 'cache_dir'))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, name)
        with open(cache_file, 'w') as cache:
            json.dump(value, cache)