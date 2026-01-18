from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DODatabaseInfo(object):

    def __init__(self, module):
        self.module = module
        self.rest = DigitalOceanHelper(module)
        self.module.params.pop('oauth_token')
        self.id = None
        self.name = None

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
                for database in json_data['databases']:
                    if database.get('name', None) == database_name:
                        self.id = database.get('id', None)
                        self.name = database.get('name', None)
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

    def get_databases(self):
        all_databases = []
        page = 1
        while page is not None:
            response = self.rest.get('databases?page={0}'.format(page))
            json_data = response.json
            if response.status_code == 200:
                databases = json_data.get('databases', None)
                if databases is not None and isinstance(databases, list):
                    all_databases.append(databases)
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return {'databases': all_databases}