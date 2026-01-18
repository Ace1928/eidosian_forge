from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _is_root_del(self, w, h, key):
    """
        This function checks whether a root attribute which can have
        further child attributes needed to be deleted.
        :param w: the desired config.
        :param h: the target config.
        :param key: attribute name.
        :return: True/False.
        """
    return True if h and key in h and (not w or key not in w or (not w[key])) else False