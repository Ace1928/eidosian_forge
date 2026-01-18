from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
@six.add_metaclass(abc.ABCMeta)
class OneViewModuleBase(object):
    MSG_CREATED = 'Resource created successfully.'
    MSG_UPDATED = 'Resource updated successfully.'
    MSG_DELETED = 'Resource deleted successfully.'
    MSG_ALREADY_PRESENT = 'Resource is already present.'
    MSG_ALREADY_ABSENT = 'Resource is already absent.'
    MSG_DIFF_AT_KEY = "Difference found at key '{0}'. "
    ONEVIEW_COMMON_ARGS = dict(config=dict(type='path'), hostname=dict(type='str'), username=dict(type='str'), password=dict(type='str', no_log=True), api_version=dict(type='int'), image_streamer_hostname=dict(type='str'))
    ONEVIEW_VALIDATE_ETAG_ARGS = dict(validate_etag=dict(type='bool', default=True))
    resource_client = None

    def __init__(self, additional_arg_spec=None, validate_etag_support=False, supports_check_mode=False):
        """
        OneViewModuleBase constructor.

        :arg dict additional_arg_spec: Additional argument spec definition.
        :arg bool validate_etag_support: Enables support to eTag validation.
        """
        argument_spec = self._build_argument_spec(additional_arg_spec, validate_etag_support)
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=supports_check_mode)
        self._check_hpe_oneview_sdk()
        self._create_oneview_client()
        self.state = self.module.params.get('state')
        self.data = self.module.params.get('data')
        self.facts_params = self.module.params.get('params') or {}
        self.options = transform_list_to_dict(self.module.params.get('options'))
        self.validate_etag_support = validate_etag_support

    def _build_argument_spec(self, additional_arg_spec, validate_etag_support):
        merged_arg_spec = dict()
        merged_arg_spec.update(self.ONEVIEW_COMMON_ARGS)
        if validate_etag_support:
            merged_arg_spec.update(self.ONEVIEW_VALIDATE_ETAG_ARGS)
        if additional_arg_spec:
            merged_arg_spec.update(additional_arg_spec)
        return merged_arg_spec

    def _check_hpe_oneview_sdk(self):
        if not HAS_HPE_ONEVIEW:
            self.module.fail_json(msg=missing_required_lib('hpOneView'), exception=HPE_ONEVIEW_IMP_ERR)

    def _create_oneview_client(self):
        if self.module.params.get('hostname'):
            config = dict(ip=self.module.params['hostname'], credentials=dict(userName=self.module.params['username'], password=self.module.params['password']), api_version=self.module.params['api_version'], image_streamer_ip=self.module.params['image_streamer_hostname'])
            self.oneview_client = OneViewClient(config)
        elif not self.module.params['config']:
            self.oneview_client = OneViewClient.from_environment_variables()
        else:
            self.oneview_client = OneViewClient.from_json_file(self.module.params['config'])

    @abc.abstractmethod
    def execute_module(self):
        """
        Abstract method, must be implemented by the inheritor.

        This method is called from the run method. It should contains the module logic

        :return: dict: It must return a dictionary with the attributes for the module result,
            such as ansible_facts, msg and changed.
        """
        pass

    def run(self):
        """
        Common implementation of the OneView run modules.

        It calls the inheritor 'execute_module' function and sends the return to the Ansible.

        It handles any OneViewModuleException in order to signal a failure to Ansible, with a descriptive error message.

        """
        try:
            if self.validate_etag_support:
                if not self.module.params.get('validate_etag'):
                    self.oneview_client.connection.disable_etag_validation()
            result = self.execute_module()
            if 'changed' not in result:
                result['changed'] = False
            self.module.exit_json(**result)
        except OneViewModuleException as exception:
            error_msg = '; '.join((to_native(e) for e in exception.args))
            self.module.fail_json(msg=error_msg, exception=traceback.format_exc())

    def resource_absent(self, resource, method='delete'):
        """
        Generic implementation of the absent state for the OneView resources.

        It checks if the resource needs to be removed.

        :arg dict resource: Resource to delete.
        :arg str method: Function of the OneView client that will be called for resource deletion.
            Usually delete or remove.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
        if resource:
            getattr(self.resource_client, method)(resource)
            return {'changed': True, 'msg': self.MSG_DELETED}
        else:
            return {'changed': False, 'msg': self.MSG_ALREADY_ABSENT}

    def get_by_name(self, name):
        """
        Generic get by name implementation.

        :arg str name: Resource name to search for.

        :return: The resource found or None.
        """
        result = self.resource_client.get_by('name', name)
        return result[0] if result else None

    def resource_present(self, resource, fact_name, create_method='create'):
        """
        Generic implementation of the present state for the OneView resources.

        It checks if the resource needs to be created or updated.

        :arg dict resource: Resource to create or update.
        :arg str fact_name: Name of the fact returned to the Ansible.
        :arg str create_method: Function of the OneView client that will be called for resource creation.
            Usually create or add.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
        changed = False
        if 'newName' in self.data:
            self.data['name'] = self.data.pop('newName')
        if not resource:
            resource = getattr(self.resource_client, create_method)(self.data)
            msg = self.MSG_CREATED
            changed = True
        else:
            merged_data = resource.copy()
            merged_data.update(self.data)
            if self.compare(resource, merged_data):
                msg = self.MSG_ALREADY_PRESENT
            else:
                resource = self.resource_client.update(merged_data)
                changed = True
                msg = self.MSG_UPDATED
        return dict(msg=msg, changed=changed, ansible_facts={fact_name: resource})

    def resource_scopes_set(self, state, fact_name, scope_uris):
        """
        Generic implementation of the scopes update PATCH for the OneView resources.
        It checks if the resource needs to be updated with the current scopes.
        This method is meant to be run after ensuring the present state.
        :arg dict state: Dict containing the data from the last state results in the resource.
            It needs to have the 'msg', 'changed', and 'ansible_facts' entries.
        :arg str fact_name: Name of the fact returned to the Ansible.
        :arg list scope_uris: List with all the scope URIs to be added to the resource.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
        if scope_uris is None:
            scope_uris = []
        resource = state['ansible_facts'][fact_name]
        operation_data = dict(operation='replace', path='/scopeUris', value=scope_uris)
        if resource['scopeUris'] is None or set(resource['scopeUris']) != set(scope_uris):
            state['ansible_facts'][fact_name] = self.resource_client.patch(resource['uri'], **operation_data)
            state['changed'] = True
            state['msg'] = self.MSG_UPDATED
        return state

    def compare(self, first_resource, second_resource):
        """
        Recursively compares dictionary contents equivalence, ignoring types and elements order.
        Particularities of the comparison:
            - Inexistent key = None
            - These values are considered equal: None, empty, False
            - Lists are compared value by value after a sort, if they have same size.
            - Each element is converted to str before the comparison.
        :arg dict first_resource: first dictionary
        :arg dict second_resource: second dictionary
        :return: bool: True when equal, False when different.
        """
        resource1 = first_resource
        resource2 = second_resource
        debug_resources = 'resource1 = {0}, resource2 = {1}'.format(resource1, resource2)
        if resource1 and (not resource2):
            self.module.log('resource1 and not resource2. ' + debug_resources)
            return False
        for key in resource1:
            if key not in resource2:
                if resource1[key] is not None:
                    self.module.log(self.MSG_DIFF_AT_KEY.format(key) + debug_resources)
                    return False
            elif not resource1[key] and (not resource2[key]):
                continue
            elif isinstance(resource1[key], Mapping):
                if not self.compare(resource1[key], resource2[key]):
                    self.module.log(self.MSG_DIFF_AT_KEY.format(key) + debug_resources)
                    return False
            elif isinstance(resource1[key], list):
                if not self.compare_list(resource1[key], resource2[key]):
                    self.module.log(self.MSG_DIFF_AT_KEY.format(key) + debug_resources)
                    return False
            elif _standardize_value(resource1[key]) != _standardize_value(resource2[key]):
                self.module.log(self.MSG_DIFF_AT_KEY.format(key) + debug_resources)
                return False
        for key in resource2.keys():
            if key not in resource1:
                if resource2[key] is not None:
                    self.module.log(self.MSG_DIFF_AT_KEY.format(key) + debug_resources)
                    return False
        return True

    def compare_list(self, first_resource, second_resource):
        """
        Recursively compares lists contents equivalence, ignoring types and element orders.
        Lists with same size are compared value by value after a sort,
        each element is converted to str before the comparison.
        :arg list first_resource: first list
        :arg list second_resource: second list
        :return: True when equal; False when different.
        """
        resource1 = first_resource
        resource2 = second_resource
        debug_resources = 'resource1 = {0}, resource2 = {1}'.format(resource1, resource2)
        if not resource2:
            self.module.log('resource 2 is null. ' + debug_resources)
            return False
        if len(resource1) != len(resource2):
            self.module.log('resources have different length. ' + debug_resources)
            return False
        resource1 = sorted(resource1, key=_str_sorted)
        resource2 = sorted(resource2, key=_str_sorted)
        for i, val in enumerate(resource1):
            if isinstance(val, Mapping):
                if not self.compare(val, resource2[i]):
                    self.module.log('resources are different. ' + debug_resources)
                    return False
            elif isinstance(val, list):
                if not self.compare_list(val, resource2[i]):
                    self.module.log('lists are different. ' + debug_resources)
                    return False
            elif _standardize_value(val) != _standardize_value(resource2[i]):
                self.module.log('values are different. ' + debug_resources)
                return False
        return True