from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOProject(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.module.params.pop('oauth_token')
        self.id = None
        self.name = None
        self.purpose = None
        self.description = None
        self.environment = None
        self.is_default = None

    def get_by_id(self, project_id):
        if not project_id:
            return None
        response = self.rest.get('projects/{0}'.format(project_id))
        json_data = response.json
        if response.status_code == 200:
            project = json_data.get('project', None)
            if project is not None:
                self.id = project.get('id', None)
                self.name = project.get('name', None)
                self.purpose = project.get('purpose', None)
                self.description = project.get('description', None)
                self.environment = project.get('environment', None)
                self.is_default = project.get('is_default', None)
            return json_data
        return None

    def get_by_name(self, project_name):
        if not project_name:
            return None
        page = 1
        while page is not None:
            response = self.rest.get('projects?page={0}'.format(page))
            json_data = response.json
            if response.status_code == 200:
                for project in json_data['projects']:
                    if project.get('name', None) == project_name:
                        self.id = project.get('id', None)
                        self.name = project.get('name', None)
                        self.description = project.get('description', None)
                        self.purpose = project.get('purpose', None)
                        self.environment = project.get('environment', None)
                        self.is_default = project.get('is_default', None)
                        return {'project': project}
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return None

    def get_project(self):
        json_data = self.get_by_id(self.module.params['id'])
        if not json_data:
            json_data = self.get_by_name(self.module.params['name'])
        return json_data

    def create(self, state):
        json_data = self.get_project()
        request_params = dict(self.module.params)
        if json_data is not None:
            changed = False
            valid_purpose = ['Just trying out DigitalOcean', 'Class project/Educational Purposes', 'Website or blog', 'Web Application', 'Service or API', 'Mobile Application', 'Machine Learning/AI/Data Processing', 'IoT', 'Operational/Developer tooling']
            for key in request_params.keys():
                if key == 'purpose' and request_params[key] is not None and (request_params[key] not in valid_purpose):
                    param = 'Other: ' + request_params[key]
                else:
                    param = request_params[key]
                if json_data['project'][key] != param and param is not None:
                    changed = True
            if changed:
                response = self.rest.put('projects/{0}'.format(json_data['project']['id']), data=request_params)
                if response.status_code != 200:
                    self.module.fail_json(changed=False, msg='Unable to update project')
                self.module.exit_json(changed=True, data=response.json)
            else:
                self.module.exit_json(changed=False, data=json_data)
        else:
            response = self.rest.post('projects', data=request_params)
            if response.status_code != 201:
                self.module.fail_json(changed=False, msg='Unable to create project')
            self.module.exit_json(changed=True, data=response.json)

    def delete(self):
        json_data = self.get_project()
        if json_data:
            if self.module.check_mode:
                self.module.exit_json(changed=True)
            response = self.rest.delete('projects/{0}'.format(json_data['project']['id']))
            json_data = response.json
            if response.status_code == 204:
                self.module.exit_json(changed=True, msg='Project deleted')
            self.module.fail_json(changed=False, msg='Failed to delete project')
        else:
            self.module.exit_json(changed=False, msg='Project not found')