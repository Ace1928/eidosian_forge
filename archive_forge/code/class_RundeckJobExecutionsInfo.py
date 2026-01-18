from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.rundeck import (
class RundeckJobExecutionsInfo(object):

    def __init__(self, module):
        self.module = module
        self.url = self.module.params['url']
        self.api_version = self.module.params['api_version']
        self.job_id = self.module.params['job_id']
        self.offset = self.module.params['offset']
        self.max = self.module.params['max']
        self.status = self.module.params['status'] or ''

    def job_executions(self):
        response, info = api_request(module=self.module, endpoint='job/%s/executions?offset=%s&max=%s&status=%s' % (quote(self.job_id), self.offset, self.max, self.status), method='GET')
        if info['status'] != 200:
            self.module.fail_json(msg=info['msg'], executions=response)
        self.module.exit_json(msg='Executions info result', **response)