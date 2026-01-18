from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_volume(self):
    volume = self.get_volume()
    if volume:
        if 'attached' in volume and (not self.module.params.get('force')):
            self.module.fail_json(msg="Volume '%s' is attached, use force=true for detaching and removing the volume." % volume.get('name'))
        self.result['changed'] = True
        if not self.module.check_mode:
            volume = self.detached_volume()
            res = self.query_api('deleteVolume', id=volume['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'volume')
    return volume