from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class ConnectorConfig(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(additional_properties=params.get('additionalProperties'), attributes=params.get('attributes'), connector_name=params.get('connectorName'), connector_type=params.get('connectorType'), deltasync_schedule=params.get('deltasyncSchedule'), description=params.get('description'), enabled=params.get('enabled'), fullsync_schedule=params.get('fullsyncSchedule'), protocol=params.get('protocol'), skip_certificate_validations=params.get('skipCertificateValidations'), url=params.get('url'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='edda', function='get_connector_config_by_connector_name', params={'connector_name': name}, handle_func_exception=False).response['response']
            result = get_dict_result(result, 'connectorName', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        name = self.new_object.get('connectorName')
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
        obj_params = [('additionalProperties', 'additional_properties'), ('attributes', 'attributes'), ('connectorName', 'connector_name'), ('connectorType', 'connector_type'), ('deltasyncSchedule', 'deltasync_schedule'), ('description', 'description'), ('enabled', 'enabled'), ('fullsyncSchedule', 'fullsync_schedule'), ('protocol', 'protocol'), ('skipCertificateValidations', 'skip_certificate_validations'), ('url', 'url')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='edda', function='create_connector_config', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not name:
            name_ = self.get_object_by_id(id).get('name')
            self.new_object.update(dict(name=name_))
        result = self.ise.exec(family='edda', function='update_connector_config_by_connector_name', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not name:
            name_ = self.get_object_by_id(id).get('name')
            self.new_object.update(dict(name=name_))
        result = self.ise.exec(family='edda', function='delete_connector_config_by_connector_name', params=self.new_object).response
        return result