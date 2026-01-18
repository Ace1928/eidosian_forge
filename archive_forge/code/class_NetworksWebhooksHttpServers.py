from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksWebhooksHttpServers(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), url=params.get('url'), sharedSecret=params.get('sharedSecret'), payloadTemplate=params.get('payloadTemplate'), networkId=params.get('networkId'), httpServerId=params.get('httpServerId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('httpServerId') is not None or self.new_object.get('http_server_id') is not None:
            new_object_params['httpServerId'] = self.new_object.get('httpServerId') or self.new_object.get('http_server_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('url') is not None or self.new_object.get('url') is not None:
            new_object_params['url'] = self.new_object.get('url') or self.new_object.get('url')
        if self.new_object.get('sharedSecret') is not None or self.new_object.get('shared_secret') is not None:
            new_object_params['sharedSecret'] = self.new_object.get('sharedSecret') or self.new_object.get('shared_secret')
        if self.new_object.get('payloadTemplate') is not None or self.new_object.get('payload_template') is not None:
            new_object_params['payloadTemplate'] = self.new_object.get('payloadTemplate') or self.new_object.get('payload_template')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('httpServerId') is not None or self.new_object.get('http_server_id') is not None:
            new_object_params['httpServerId'] = self.new_object.get('httpServerId') or self.new_object.get('http_server_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('sharedSecret') is not None or self.new_object.get('shared_secret') is not None:
            new_object_params['sharedSecret'] = self.new_object.get('sharedSecret') or self.new_object.get('shared_secret')
        if self.new_object.get('payloadTemplate') is not None or self.new_object.get('payload_template') is not None:
            new_object_params['payloadTemplate'] = self.new_object.get('payloadTemplate') or self.new_object.get('payload_template')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('httpServerId') is not None or self.new_object.get('http_server_id') is not None:
            new_object_params['httpServerId'] = self.new_object.get('httpServerId') or self.new_object.get('http_server_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='networks', function='getNetworkWebhooksHttpServers', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='networks', function='getNetworkWebhooksHttpServer', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'httpServerId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('http_server_id') or self.new_object.get('httpServerId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('httpServerId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(httpServerId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('url', 'url'), ('sharedSecret', 'sharedSecret'), ('payloadTemplate', 'payloadTemplate'), ('networkId', 'networkId'), ('httpServerId', 'httpServerId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='networks', function='createNetworkWebhooksHttpServer', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('httpServerId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('httpServerId')
            if id_:
                self.new_object.update(dict(httpServerId=id_))
        result = self.meraki.exec_meraki(family='networks', function='updateNetworkWebhooksHttpServer', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('httpServerId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('httpServerId')
            if id_:
                self.new_object.update(dict(httpServerId=id_))
        result = self.meraki.exec_meraki(family='networks', function='deleteNetworkWebhooksHttpServer', params=self.delete_by_id_params())
        return result