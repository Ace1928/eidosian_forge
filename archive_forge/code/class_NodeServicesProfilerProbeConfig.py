from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class NodeServicesProfilerProbeConfig(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(active_directory=params.get('activeDirectory'), dhcp=params.get('dhcp'), dhcp_span=params.get('dhcpSpan'), dns=params.get('dns'), http=params.get('http'), netflow=params.get('netflow'), nmap=params.get('nmap'), pxgrid=params.get('pxgrid'), radius=params.get('radius'), snmp_query=params.get('snmpQuery'), snmp_trap=params.get('snmpTrap'), hostname=params.get('hostname'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='node_services', function='get_profiler_probe_config', params={'hostname': name}, handle_func_exception=False).response['response']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        name = self.new_object.get('hostname')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('activeDirectory', 'active_directory', False), ('dhcp', 'dhcp', False), ('dhcpSpan', 'dhcp_span', False), ('dns', 'dns', False), ('http', 'http', False), ('netflow', 'netflow', False), ('nmap', 'nmap', False), ('pxgrid', 'pxgrid', False), ('radius', 'radius', False), ('snmpQuery', 'snmp_query', False), ('snmpTrap', 'snmp_trap', False), ('hostname', 'hostname', True)]
        return any((not ise_compare_equality2(current_obj.get(ise_param), requested_obj.get(ansible_param), is_query_param) for ise_param, ansible_param, is_query_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('hostname')
        result = None
        if not name:
            name_ = self.get_object_by_id(id).get('hostname')
            self.new_object.update(dict(name=name_))
        result = self.ise.exec(family='node_services', function='set_profiler_probe_config', params=self.new_object).response
        return result