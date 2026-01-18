from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
class ImageFacts(object):

    def __init__(self, module):
        self.module = module
        self.filters = module.params['filters']

    def return_all_installed_images(self):
        cmd = [self.module.get_bin_path('imgadm'), 'list', '-j']
        if self.filters:
            cmd.append(self.filters)
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.exit_json(msg='Failed to get all installed images', stderr=err)
        images = json.loads(out)
        result = {}
        for image in images:
            result[image['manifest']['uuid']] = image['manifest']
            for attrib in ['clones', 'source', 'zpool']:
                result[image['manifest']['uuid']][attrib] = image[attrib]
        return result