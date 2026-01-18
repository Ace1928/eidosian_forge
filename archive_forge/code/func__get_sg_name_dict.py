import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_sg_name_dict(self, data, page_size, no_nameconv):
    """Get names of security groups referred in the retrieved rules.

        :return: a dict from secgroup ID to secgroup name
        """
    if no_nameconv:
        return {}
    neutron_client = self.get_client()
    search_opts = {'fields': ['id', 'name']}
    if self.pagination_support:
        if page_size:
            search_opts.update({'limit': page_size})
    sec_group_ids = set()
    for rule in data:
        for key in self.replace_rules:
            if rule.get(key):
                sec_group_ids.add(rule[key])
    sec_group_ids = list(sec_group_ids)

    def _get_sec_group_list(sec_group_ids):
        search_opts['id'] = sec_group_ids
        return neutron_client.list_security_groups(**search_opts).get('security_groups', [])
    try:
        secgroups = _get_sec_group_list(sec_group_ids)
    except exceptions.RequestURITooLong as uri_len_exc:
        sec_group_id_filter_len = 40
        sec_group_count = len(sec_group_ids)
        max_size = sec_group_id_filter_len * sec_group_count - uri_len_exc.excess
        chunk_size = max_size // sec_group_id_filter_len
        secgroups = []
        for i in range(0, sec_group_count, chunk_size):
            secgroups.extend(_get_sec_group_list(sec_group_ids[i:i + chunk_size]))
    return dict([(sg['id'], sg['name']) for sg in secgroups if sg['name']])