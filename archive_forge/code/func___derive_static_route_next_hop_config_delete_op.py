from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def __derive_static_route_next_hop_config_delete_op(key_set, command, exist_conf):
    new_conf = []
    if is_delete_all:
        return (True, new_conf)
    metric = command.get('metric', None)
    tag = command.get('tag', None)
    track = command.get('track', None)
    if metric is None and tag is None and (track is None):
        return (True, new_conf)
    new_conf = exist_conf
    conf_metric = new_conf.get('metric', None)
    conf_tag = new_conf.get('tag', None)
    conf_track = new_conf.get('track', None)
    if metric == conf_metric:
        new_conf['metric'] = None
    if tag == conf_tag:
        new_conf['tag'] = None
    if track == conf_track:
        new_conf['track'] = None
    return (True, new_conf)