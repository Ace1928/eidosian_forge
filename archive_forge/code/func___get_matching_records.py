from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __get_matching_records(self):
    """Collect exact and similar records

        It returns an exact record if there is any match along with the record_id.
        It also returns multiple records if there is no exact match
        """
    for record in self.records:
        r = dict(record)
        del r['id']
        if r == self.payload:
            return (r, record['id'], None)
    similar_records = []
    for record in self.records:
        if record['type'] == self.payload['type'] and record['name'] == self.payload['name']:
            similar_records.append(record)
    if similar_records:
        return (None, None, similar_records)
    return (None, None, None)