from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.rundeck import (
class RundeckJobRun(object):

    def __init__(self, module):
        self.module = module
        self.url = self.module.params['url']
        self.api_version = self.module.params['api_version']
        self.job_id = self.module.params['job_id']
        self.job_options = self.module.params['job_options'] or {}
        self.filter_nodes = self.module.params['filter_nodes'] or ''
        self.run_at_time = self.module.params['run_at_time'] or ''
        self.loglevel = self.module.params['loglevel'].upper()
        self.wait_execution = self.module.params['wait_execution']
        self.wait_execution_delay = self.module.params['wait_execution_delay']
        self.wait_execution_timeout = self.module.params['wait_execution_timeout']
        self.abort_on_timeout = self.module.params['abort_on_timeout']
        for k, v in self.job_options.items():
            if not isinstance(v, str):
                self.module.exit_json(msg="Job option '%s' value must be a string" % k, execution_info={})

    def job_status_check(self, execution_id):
        response = dict()
        timeout = False
        due = datetime.now() + timedelta(seconds=self.wait_execution_timeout)
        while not timeout:
            endpoint = 'execution/%d' % execution_id
            response = api_request(module=self.module, endpoint=endpoint)[0]
            output = api_request(module=self.module, endpoint='execution/%d/output' % execution_id)
            log_output = '\n'.join([x['log'] for x in output[0]['entries']])
            response.update({'output': log_output})
            if response['status'] == 'aborted':
                break
            elif response['status'] == 'scheduled':
                self.module.exit_json(msg='Job scheduled to run at %s' % self.run_at_time, execution_info=response, changed=True)
            elif response['status'] == 'failed':
                self.module.fail_json(msg='Job execution failed', execution_info=response)
            elif response['status'] == 'succeeded':
                self.module.exit_json(msg='Job execution succeeded!', execution_info=response)
            if datetime.now() >= due:
                timeout = True
                break
            sleep(self.wait_execution_delay)
        response.update({'timed_out': timeout})
        return response

    def job_run(self):
        response, info = api_request(module=self.module, endpoint='job/%s/run' % quote(self.job_id), method='POST', data={'loglevel': self.loglevel, 'options': self.job_options, 'runAtTime': self.run_at_time, 'filter': self.filter_nodes})
        if info['status'] != 200:
            self.module.fail_json(msg=info['msg'])
        if not self.wait_execution:
            self.module.exit_json(msg='Job run send successfully!', execution_info=response)
        job_status = self.job_status_check(response['id'])
        if job_status['timed_out']:
            if self.abort_on_timeout:
                api_request(module=self.module, endpoint='execution/%s/abort' % response['id'], method='GET')
                abort_status = self.job_status_check(response['id'])
                self.module.fail_json(msg='Job execution aborted due the timeout specified', execution_info=abort_status)
            self.module.fail_json(msg='Job execution timed out', execution_info=job_status)