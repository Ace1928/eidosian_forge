from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackProject(AnsibleCloudStack):

    def get_project(self):
        if not self.project:
            project = self.module.params.get('name')
            args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'fetch_list': True}
            projects = self.query_api('listProjects', **args)
            if projects:
                for p in projects:
                    if project.lower() in [p['name'].lower(), p['id']]:
                        self.project = p
                        break
        return self.project

    def present_project(self):
        project = self.get_project()
        if not project:
            project = self.create_project(project)
        else:
            project = self.update_project(project)
        if project:
            project = self.ensure_tags(resource=project, resource_type='project')
            self.project = project
        return project

    def update_project(self, project):
        args = {'id': project['id'], 'displaytext': self.get_or_fallback('display_text', 'name')}
        if self.has_changed(args, project):
            self.result['changed'] = True
            if not self.module.check_mode:
                project = self.query_api('updateProject', **args)
                poll_async = self.module.params.get('poll_async')
                if project and poll_async:
                    project = self.poll_job(project, 'project')
        return project

    def create_project(self, project):
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'account': self.get_account('name'), 'domainid': self.get_domain('id')}
        if not self.module.check_mode:
            project = self.query_api('createProject', **args)
            poll_async = self.module.params.get('poll_async')
            if project and poll_async:
                project = self.poll_job(project, 'project')
        return project

    def state_project(self, state='active'):
        project = self.present_project()
        if project['state'].lower() != state:
            self.result['changed'] = True
            args = {'id': project['id']}
            if not self.module.check_mode:
                if state == 'suspended':
                    project = self.query_api('suspendProject', **args)
                else:
                    project = self.query_api('activateProject', **args)
                poll_async = self.module.params.get('poll_async')
                if project and poll_async:
                    project = self.poll_job(project, 'project')
        return project

    def absent_project(self):
        project = self.get_project()
        if project:
            self.result['changed'] = True
            args = {'id': project['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteProject', **args)
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    res = self.poll_job(res, 'project')
            return project