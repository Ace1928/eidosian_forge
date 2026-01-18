from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
class JenkinsBuildInfo:

    def __init__(self, module):
        self.module = module
        self.name = module.params.get('name')
        self.password = module.params.get('password')
        self.token = module.params.get('token')
        self.user = module.params.get('user')
        self.jenkins_url = module.params.get('url')
        self.build_number = module.params.get('build_number')
        self.server = self.get_jenkins_connection()
        self.result = {'changed': False, 'url': self.jenkins_url, 'name': self.name, 'user': self.user}

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
            self.module.fail_json(msg='Unable to connect to Jenkins server, %s' % to_native(e))

    def get_build_status(self):
        try:
            if self.build_number is None:
                job_info = self.server.get_job_info(self.name)
                self.build_number = job_info['lastBuild']['number']
            return self.server.get_build_info(self.name, self.build_number)
        except jenkins.JenkinsException as e:
            response = {}
            response['result'] = 'ABSENT'
            return response
        except Exception as e:
            self.module.fail_json(msg='Unable to fetch build information, %s' % to_native(e), exception=traceback.format_exc())

    def get_result(self):
        result = self.result
        build_status = self.get_build_status()
        if build_status['result'] == 'ABSENT':
            result['failed'] = True
        result['build_info'] = build_status
        return result