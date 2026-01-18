from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_selements(self, graph, nodes, icon_ids):
    selements = []
    icon_sizes = {}
    scales = self._get_scales(graph)
    for selementid, (node, data) in enumerate(nodes.items(), start=1):
        selement = {'selementid': selementid}
        data['selementid'] = selementid
        images_info = self._get_images_info(data, icon_ids)
        selement.update(images_info)
        image_id = images_info['iconid_off']
        if image_id not in icon_sizes:
            icon_sizes[image_id] = self._get_icon_size(image_id)
        pos = self._convert_coordinates(data['pos'], scales, icon_sizes[image_id])
        selement.update(pos)
        selement['label'] = remove_quotes(node)
        element_type = self._get_element_type(data)
        selement.update(element_type)
        label = self._get_label(data)
        if label:
            selement['label'] = label
        urls = self._get_urls(data)
        if urls:
            selement['urls'] = urls
        selements.append(selement)
    return selements