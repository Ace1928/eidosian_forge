from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOVPC(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.module.params.pop('oauth_token')
        self.name = module.params.get('name', None)
        self.description = module.params.get('description', None)
        self.default = module.params.get('default', False)
        self.region = module.params.get('region', None)
        self.ip_range = module.params.get('ip_range', None)
        self.vpc_id = module.params.get('vpc_id', None)

    def get_by_name(self):
        page = 1
        while page is not None:
            response = self.rest.get('vpcs?page={0}'.format(page))
            json_data = response.json
            if response.status_code == 200:
                for vpc in json_data['vpcs']:
                    if vpc.get('name', None) == self.name:
                        return vpc
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return None

    def create(self):
        if self.module.check_mode:
            return self.module.exit_json(changed=True)
        vpc = self.get_by_name()
        if vpc is not None:
            vpc_id = vpc.get('id', None)
            if vpc_id is not None:
                data = {'name': self.name}
                if self.description is not None:
                    data['description'] = self.description
                if self.default is not False:
                    data['default'] = True
                response = self.rest.put('vpcs/{0}'.format(vpc_id), data=data)
                json = response.json
                if response.status_code != 200:
                    self.module.fail_json(msg='Failed to update VPC {0} in {1}: {2}'.format(self.name, self.region, json['message']))
                else:
                    self.module.exit_json(changed=False, data=json, msg='Updated VPC {0} in {1}'.format(self.name, self.region))
            else:
                self.module.fail_json(changed=False, msg='Unexpected error, please file a bug')
        else:
            data = {'name': self.name, 'region': self.region}
            if self.description is not None:
                data['description'] = self.description
            if self.ip_range is not None:
                data['ip_range'] = self.ip_range
            response = self.rest.post('vpcs', data=data)
            status = response.status_code
            json = response.json
            if status == 201:
                self.module.exit_json(changed=True, data=json, msg='Created VPC {0} in {1}'.format(self.name, self.region))
            else:
                self.module.fail_json(changed=False, msg='Failed to create VPC {0} in {1}: {2}'.format(self.name, self.region, json['message']))

    def delete(self):
        if self.module.check_mode:
            return self.module.exit_json(changed=True)
        vpc = self.get_by_name()
        if vpc is None:
            self.module.fail_json(msg='Unable to find VPC {0} in {1}'.format(self.name, self.region))
        else:
            vpc_id = vpc.get('id', None)
            if vpc_id is not None:
                response = self.rest.delete('vpcs/{0}'.format(str(vpc_id)))
                status = response.status_code
                json = response.json
                if status == 204:
                    self.module.exit_json(changed=True, msg='Deleted VPC {0} in {1} ({2})'.format(self.name, self.region, vpc_id))
                else:
                    json = response.json
                    self.module.fail_json(changed=False, msg='Failed to delete VPC {0} ({1}): {2}'.format(self.name, vpc_id, json['message']))