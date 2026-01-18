from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapNetRoutes:
    """
    Create, Modifies and Destroys a Net Route
    """

    def __init__(self):
        """
        Initialize the Ontap Net Route class
        """
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), vserver=dict(required=False, type='str'), destination=dict(required=True, type='str'), gateway=dict(required=True, type='str'), metric=dict(required=False, type='int'), from_destination=dict(required=False, type='str', default=None), from_gateway=dict(required=False, type='str', default=None), from_metric=dict(required=False, type='int', default=None)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, ['from_metric'], [['metric', (9, 11, 0)]])
        self.validate_options()
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])

    def validate_options(self):
        errors = []
        example = ''
        if not self.use_rest and 'vserver' not in self.parameters:
            errors.append('vserver is a required parameter when using ZAPI')
        for attr in ('destination', 'from_destination'):
            value = self.parameters.get(attr)
            if value is not None and '/' not in value:
                errors.append("Expecting '/' in '%s'" % value)
                example = 'Examples: 10.7.125.5/20, fd20:13::/64'
        if errors:
            if example:
                errors.append(example)
            self.module.fail_json(msg='Error: %s.' % '.  '.join(errors))

    @staticmethod
    def sanitize_exception(action, exc):
        if action == 'create' and to_native(exc.code) == '13001' and ('already exists' in to_native(exc.message)):
            return None
        if action == 'get' and to_native(exc.code) == '15661':
            return None
        return to_native(exc)

    def create_net_route(self, current=None, fail=True):
        """
        Creates a new Route
        """
        if current is None:
            current = self.parameters
        if self.use_rest:
            api = 'network/ip/routes'
            body = {'gateway': current['gateway']}
            dest = current['destination']
            if isinstance(dest, dict):
                body['destination'] = dest
            else:
                dest = current['destination'].split('/')
                body['destination'] = {'address': dest[0], 'netmask': dest[1]}
            if current.get('vserver') is not None:
                body['svm.name'] = current['vserver']
            if current.get('metric') is not None:
                body['metric'] = current['metric']
            __, error = rest_generic.post_async(self.rest_api, api, body)
        else:
            route_obj = netapp_utils.zapi.NaElement('net-routes-create')
            route_obj.add_new_child('destination', current['destination'])
            route_obj.add_new_child('gateway', current['gateway'])
            metric = current.get('metric')
            if metric is not None:
                route_obj.add_new_child('metric', str(metric))
            try:
                self.server.invoke_successfully(route_obj, True)
                error = None
            except netapp_utils.zapi.NaApiError as exc:
                error = self.sanitize_exception('create', exc)
        if error:
            error = 'Error creating net route: %s' % error
            if fail:
                self.module.fail_json(msg=error)
        return error

    def delete_net_route(self, current):
        """
        Deletes a given Route
        """
        if self.use_rest:
            uuid = current['uuid']
            api = 'network/ip/routes'
            dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
            if error:
                self.module.fail_json(msg='Error deleting net route - %s' % error)
        else:
            route_obj = netapp_utils.zapi.NaElement('net-routes-destroy')
            route_obj.add_new_child('destination', current['destination'])
            route_obj.add_new_child('gateway', current['gateway'])
            try:
                self.server.invoke_successfully(route_obj, True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error deleting net route: %s' % to_native(error), exception=traceback.format_exc())

    def recreate_net_route(self, current):
        """
        Modify a net route
        Since we cannot modify a route, we are deleting the existing route, and creating a new one.
        """
        self.delete_net_route(current)
        if current.get('metric') is not None and self.parameters.get('metric') is None:
            self.parameters['metric'] = current['metric']
        error = self.create_net_route(fail=False)
        if error:
            self.create_net_route(current)
            self.module.fail_json(msg='Error modifying net route: %s' % error, exception=traceback.format_exc())

    def get_net_route(self, params=None):
        """
        Checks to see if a route exist or not
        :return: NaElement object if a route exists, None otherwise
        """
        if params is None:
            params = self.parameters
        if self.use_rest:
            api = 'network/ip/routes'
            fields = 'destination,gateway,svm,scope'
            if self.parameters.get('metric') is not None:
                fields += ',metric'
            query = {'destination.address': params['destination'].split('/')[0], 'gateway': params['gateway']}
            if params.get('vserver') is None:
                query['scope'] = 'cluster'
            else:
                query['scope'] = 'svm'
                query['svm.name'] = params['vserver']
            record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
            if error:
                self.module.fail_json(msg='Error fetching net route: %s' % error)
            if record and 'metric' not in record:
                record['metric'] = None
            return record
        else:
            route_obj = netapp_utils.zapi.NaElement('net-routes-get')
            for attr in ('destination', 'gateway'):
                route_obj.add_new_child(attr, params[attr])
            try:
                result = self.server.invoke_successfully(route_obj, True)
            except netapp_utils.zapi.NaApiError as exc:
                error = self.sanitize_exception('get', exc)
                if error is None:
                    return None
                self.module.fail_json(msg='Error fetching net route: %s' % error, exception=traceback.format_exc())
            if result.get_child_by_name('attributes') is not None:
                route_info = result.get_child_by_name('attributes').get_child_by_name('net-vs-routes-info')
                return {'destination': route_info.get_child_content('destination'), 'gateway': route_info.get_child_content('gateway'), 'metric': int(route_info.get_child_content('metric'))}
            return None

    def apply(self):
        """
        Run Module based on play book
        """
        modify, rename = (False, False)
        current = self.get_net_route()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and any((self.parameters.get(attr) is not None for attr in ('from_gateway', 'from_destination'))):
            from_params = {'gateway': self.parameters.get('from_gateway', self.parameters['gateway']), 'destination': self.parameters.get('from_destination', self.parameters['destination'])}
            if self.parameters.get('vserver'):
                from_params['vserver'] = self.parameters['vserver']
            current = self.get_net_route(from_params)
            if current is None:
                self.module.fail_json(msg='Error modifying: route %s does not exist' % self.parameters['from_destination'])
            rename = True
            cd_action = None
        if cd_action is None and self.parameters.get('metric') is not None and current:
            modify = self.parameters['metric'] != current['metric']
            if modify:
                self.na_helper.changed = True
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_net_route()
            elif cd_action == 'delete':
                self.delete_net_route(current)
            elif rename or modify:
                self.recreate_net_route(current)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify, extra_responses={'rename': rename})
        self.module.exit_json(**result)