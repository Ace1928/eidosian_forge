from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
class RhsmPools(object):
    """
    This class is used for manipulating pools subscriptions with RHSM

    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 9.0.0.
    There is no replacement for it; please contact the community.general
    maintainers in case you are using it.
    """

    def __init__(self, module):
        self.module = module
        self.products = self._load_product_list()
        self.module.deprecate('The RhsmPools class is deprecated with no replacement.', version='9.0.0', collection_name='community.general')

    def __iter__(self):
        return self.products.__iter__()

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

    def filter(self, regexp='^$'):
        """
            Return a list of RhsmPools whose name matches the provided regular expression
        """
        r = re.compile(regexp)
        for product in self.products:
            if r.search(product._name):
                yield product