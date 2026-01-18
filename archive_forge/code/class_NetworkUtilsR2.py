import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
class NetworkUtilsR2(NetworkUtils):
    _PORT_EXT_ACL_SET_DATA = 'Msvm_EthernetSwitchPortExtendedAclSettingData'
    _MAX_WEIGHT = 65500
    _REJECT_ACLS_COUNT = 16

    def _create_security_acl(self, sg_rule, weight):
        acl = super(NetworkUtilsR2, self)._create_security_acl(sg_rule, weight)
        acl.Weight = weight
        sg_rule.Weight = weight
        return acl

    def _get_new_weights(self, sg_rules, existent_acls):
        sg_rule = sg_rules[0]
        num_rules = len(sg_rules)
        existent_acls = [a for a in existent_acls if a.Action == sg_rule.Action]
        if not existent_acls:
            if sg_rule.Action == self._ACL_ACTION_DENY:
                return list(range(1, 1 + num_rules))
            else:
                return list(range(self._MAX_WEIGHT - 1, self._MAX_WEIGHT - 1 - num_rules, -1))
        weights = [a.Weight for a in existent_acls]
        if sg_rule.Action == self._ACL_ACTION_DENY:
            return [i for i in list(range(1, self._REJECT_ACLS_COUNT + 1)) if i not in weights][:num_rules]
        min_weight = min(weights)
        last_weight = min_weight - num_rules - 1
        if last_weight > self._REJECT_ACLS_COUNT:
            return list(range(min_weight - 1, last_weight, -1))
        current_weight = self._MAX_WEIGHT - 1
        new_weights = []
        for i in list(range(num_rules)):
            while current_weight in weights:
                current_weight -= 1
            new_weights.append(current_weight)
        return new_weights