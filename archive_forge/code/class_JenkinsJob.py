from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
class JenkinsJob(object):

    def __init__(self, module):
        self.module = module
        self.config = module.params.get('config')
        self.name = module.params.get('name')
        self.password = module.params.get('password')
        self.state = module.params.get('state')
        self.enabled = module.params.get('enabled')
        self.token = module.params.get('token')
        self.user = module.params.get('user')
        self.jenkins_url = module.params.get('url')
        self.server = self.get_jenkins_connection()
        self.result = {'changed': False, 'url': self.jenkins_url, 'name': self.name, 'user': self.user, 'state': self.state, 'diff': {'before': '', 'after': ''}}
        self.EXCL_STATE = 'excluded state'
        if not module.params['validate_certs']:
            os.environ['PYTHONHTTPSVERIFY'] = '0'

    def get_jenkins_connection(self):
        try:
            if self.user and self.password:
                return jenkins.Jenkins(self.jenkins_url, self.user, self.password)
            elif self.user and self.token:
                return jenkins.Jenkins(self.jenkins_url, self.user, self.token)
            elif self.user and (not (self.password or self.token)):
                return jenkins.Jenkins(self.jenkins_url, self.user)
            else:
                return jenkins.Jenkins(self.jenkins_url)
        except Exception as e:
            self.module.fail_json(msg='Unable to connect to Jenkins server, %s' % to_native(e), exception=traceback.format_exc())

    def get_job_status(self):
        try:
            response = self.server.get_job_info(self.name)
            if 'color' not in response:
                return self.EXCL_STATE
            else:
                return to_native(response['color'])
        except Exception as e:
            self.module.fail_json(msg='Unable to fetch job information, %s' % to_native(e), exception=traceback.format_exc())

    def job_exists(self):
        try:
            return bool(self.server.job_exists(self.name))
        except Exception as e:
            self.module.fail_json(msg='Unable to validate if job exists, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())

    def get_config(self):
        return job_config_to_string(self.config)

    def get_current_config(self):
        return job_config_to_string(self.server.get_job_config(self.name).encode('utf-8'))

    def has_config_changed(self):
        if self.config is None:
            return False
        config_file = self.get_config()
        machine_file = self.get_current_config()
        self.result['diff']['after'] = config_file
        self.result['diff']['before'] = machine_file
        if machine_file != config_file:
            return True
        return False

    def present_job(self):
        if self.config is None and self.enabled is None:
            self.module.fail_json(msg='one of the following params is required on state=present: config,enabled')
        if not self.job_exists():
            self.create_job()
        else:
            self.update_job()

    def has_state_changed(self, status):
        if self.enabled is None:
            return False
        return self.enabled is False and status != 'disabled' or (self.enabled is True and status == 'disabled')

    def switch_state(self):
        if self.enabled is False:
            self.server.disable_job(self.name)
        else:
            self.server.enable_job(self.name)

    def update_job(self):
        try:
            status = self.get_job_status()
            if self.has_config_changed():
                self.result['changed'] = True
                if not self.module.check_mode:
                    self.server.reconfig_job(self.name, self.get_config())
            elif status != self.EXCL_STATE and self.has_state_changed(status):
                self.result['changed'] = True
                if not self.module.check_mode:
                    self.switch_state()
        except Exception as e:
            self.module.fail_json(msg='Unable to reconfigure job, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())

    def create_job(self):
        if self.config is None:
            self.module.fail_json(msg='missing required param: config')
        self.result['changed'] = True
        try:
            config_file = self.get_config()
            self.result['diff']['after'] = config_file
            if not self.module.check_mode:
                self.server.create_job(self.name, config_file)
        except Exception as e:
            self.module.fail_json(msg='Unable to create job, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())

    def absent_job(self):
        if self.job_exists():
            self.result['changed'] = True
            self.result['diff']['before'] = self.get_current_config()
            if not self.module.check_mode:
                try:
                    self.server.delete_job(self.name)
                except Exception as e:
                    self.module.fail_json(msg='Unable to delete job, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())

    def get_result(self):
        result = self.result
        if self.job_exists():
            result['enabled'] = self.get_job_status() != 'disabled'
        else:
            result['enabled'] = None
        return result