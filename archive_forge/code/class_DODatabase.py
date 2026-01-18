from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DODatabase(object):

    def __init__(self, module):
        self.module = module
        self.rest = DigitalOceanHelper(module)
        if self.module.params.get('project_name'):
            self.projects = DigitalOceanProjects(module, self.rest)
        self.wait = self.module.params.pop('wait', True)
        self.wait_timeout = self.module.params.pop('wait_timeout', 600)
        self.module.params.pop('oauth_token')
        self.id = None
        self.name = None
        self.engine = None
        self.version = None
        self.num_nodes = None
        self.region = None
        self.status = None
        self.size = None

    def get_by_id(self, database_id):
        if database_id is None:
            return None
        response = self.rest.get('databases/{0}'.format(database_id))
        json_data = response.json
        if response.status_code == 200:
            database = json_data.get('database', None)
            if database is not None:
                self.id = database.get('id', None)
                self.name = database.get('name', None)
                self.engine = database.get('engine', None)
                self.version = database.get('version', None)
                self.num_nodes = database.get('num_nodes', None)
                self.region = database.get('region', None)
                self.status = database.get('status', None)
                self.size = database.get('size', None)
            return json_data
        return None

    def get_by_name(self, database_name):
        if database_name is None:
            return None
        page = 1
        while page is not None:
            response = self.rest.get('databases?page={0}'.format(page))
            json_data = response.json
            if response.status_code == 200:
                databases = json_data.get('databases', None)
                if databases is None or not isinstance(databases, list):
                    return None
                for database in databases:
                    if database.get('name', None) == database_name:
                        self.id = database.get('id', None)
                        self.name = database.get('name', None)
                        self.engine = database.get('engine', None)
                        self.version = database.get('version', None)
                        self.status = database.get('status', None)
                        self.num_nodes = database.get('num_nodes', None)
                        self.region = database.get('region', None)
                        self.size = database.get('size', None)
                        return {'database': database}
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return None

    def get_database(self):
        json_data = self.get_by_id(self.module.params['id'])
        if not json_data:
            json_data = self.get_by_name(self.module.params['name'])
        return json_data

    def ensure_online(self, database_id):
        end_time = time.monotonic() + self.wait_timeout
        while time.monotonic() < end_time:
            response = self.rest.get('databases/{0}'.format(database_id))
            json_data = response.json
            database = json_data.get('database', None)
            if database is not None:
                status = database.get('status', None)
                if status is not None:
                    if status == 'online':
                        return json_data
            time.sleep(10)
        self.module.fail_json(msg='Waiting for database online timeout')

    def create(self):
        json_data = self.get_database()
        if json_data is not None:
            database = json_data.get('database', None)
            if database is not None:
                self.module.exit_json(changed=False, data=json_data)
            else:
                self.module.fail_json(changed=False, msg='Unexpected error, please file a bug')
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        request_params = dict(self.module.params)
        del request_params['id']
        response = self.rest.post('databases', data=request_params)
        json_data = response.json
        if response.status_code >= 400:
            self.module.fail_json(changed=False, msg=json_data['message'])
        database = json_data.get('database', None)
        if database is None:
            self.module.fail_json(changed=False, msg='Unexpected error; please file a bug https://github.com/ansible-collections/community.digitalocean/issues')
        database_id = database.get('id', None)
        if database_id is None:
            self.module.fail_json(changed=False, msg='Unexpected error; please file a bug https://github.com/ansible-collections/community.digitalocean/issues')
        if self.wait:
            json_data = self.ensure_online(database_id)
        project_name = self.module.params.get('project_name')
        if project_name:
            urn = 'do:dbaas:{0}'.format(database_id)
            assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
            self.module.exit_json(changed=True, data=json_data, msg=error_message, assign_status=assign_status, resources=resources)
        else:
            self.module.exit_json(changed=True, data=json_data)

    def delete(self):
        json_data = self.get_database()
        if json_data is not None:
            if self.module.check_mode:
                self.module.exit_json(changed=True)
            database = json_data.get('database', None)
            database_id = database.get('id', None)
            database_name = database.get('name', None)
            database_region = database.get('region', None)
            if database_id is not None:
                response = self.rest.delete('databases/{0}'.format(database_id))
                json_data = response.json
                if response.status_code == 204:
                    self.module.exit_json(changed=True, msg='Deleted database {0} ({1}) in region {2}'.format(database_name, database_id, database_region))
                self.module.fail_json(changed=False, msg='Failed to delete database {0} ({1}) in region {2}: {3}'.format(database_name, database_id, database_region, json_data['message']))
            else:
                self.module.fail_json(changed=False, msg='Unexpected error, please file a bug')
        else:
            self.module.exit_json(changed=False, msg='Database {0} in region {1} not found'.format(self.module.params['name'], self.module.params['region']))