from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
def _load_product_list(self):
    """
            Loads list of all available pools for system in data structure
        """
    args = 'subscription-manager list --available'
    rc, stdout, stderr = self.module.run_command(args, check_rc=True)
    products = []
    for line in stdout.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().replace(' ', '')
            value = value.strip()
            if key in ['ProductName', 'SubscriptionName']:
                products.append(RhsmPool(self.module, _name=value, key=value))
            elif products:
                products[-1].__setattr__(key, value)
    return products