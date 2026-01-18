from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def add_base_volumes(self, volumes):
    """Add base volume(s) to the consistency group."""
    group_id = self.get_consistency_group()['consistency_group_id']
    member_volume_request = {'volumeToCandidates': {}}
    for volume_name, volume_info in volumes.items():
        candidate = self.get_candidate(volume_name, volume_info)
        member_volume_request['volumeToCandidates'].update({volume_info['id']: candidate['candidate']['candidate']})
    try:
        rc, resp = self.request('storage-systems/%s/consistency-groups/%s/member-volumes/batch' % (self.ssid, group_id), method='POST', data=member_volume_request)
    except Exception as error:
        self.module.fail_json(msg='Failed to add reserve capacity volume! Base volumes %s. Group [%s]. Error [%s]. Array [%s].' % (', '.join([volume for volume in member_volume_request.keys()]), self.group_name, error, self.ssid))