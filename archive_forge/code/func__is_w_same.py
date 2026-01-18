from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _is_w_same(self, w, h, key):
    """
        This function checks whether the key value is same in desired and
        target config dictionary.
        :param w: base config.
        :param h: target config.
        :param key:attribute name.
        :return: True/False.
        """
    return True if h and key in h and (h[key] == w[key]) else False