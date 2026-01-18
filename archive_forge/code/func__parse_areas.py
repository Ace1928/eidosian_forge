from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_areas(areas):
    command = []
    for area in areas:
        area_cmd = 'area ' + area['area_id']
        if area.get('default_cost'):
            command.append(area_cmd + ' default-cost ' + str(area.get('default_cost')))
        elif area.get('filter'):
            command.append(area_cmd + ' ' + _parse_areas_filter(area['filter']))
        elif area.get('not_so_stubby'):
            command.append(area_cmd + ' ' + _parse_areas_filter_notsostubby(area['not_so_stubby']))
        elif area.get('nssa'):
            command.append(area_cmd + ' ' + _parse_areas_filter_nssa(area['nssa']))
        elif area.get('range'):
            command.append(area_cmd + ' ' + _parse_areas_range(area['range']))
    return command