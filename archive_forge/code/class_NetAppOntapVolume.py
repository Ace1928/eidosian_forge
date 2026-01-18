from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
class NetAppOntapVolume:
    """Class with volume operations"""

    def __init__(self):
        """Initialize module parameters"""
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), vserver=dict(required=True, type='str'), from_name=dict(required=False, type='str'), is_infinite=dict(required=False, type='bool', default=False), is_online=dict(required=False, type='bool', default=True), size=dict(type='int', default=None), size_unit=dict(default='gb', choices=['bytes', 'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'eb', 'zb', 'yb'], type='str'), sizing_method=dict(choices=['add_new_resources', 'use_existing_resources'], type='str'), aggregate_name=dict(type='str', default=None), type=dict(type='str', default=None), export_policy=dict(type='str', default=None, aliases=['policy']), junction_path=dict(type='str', default=None), space_guarantee=dict(choices=['none', 'file', 'volume'], default=None), percent_snapshot_space=dict(type='int', default=None), volume_security_style=dict(choices=['mixed', 'ntfs', 'unified', 'unix']), encrypt=dict(required=False, type='bool'), efficiency_policy=dict(required=False, type='str'), unix_permissions=dict(required=False, type='str'), group_id=dict(required=False, type='int'), user_id=dict(required=False, type='int'), snapshot_policy=dict(required=False, type='str'), aggr_list=dict(required=False, type='list', elements='str'), aggr_list_multiplier=dict(required=False, type='int'), snapdir_access=dict(required=False, type='bool'), atime_update=dict(required=False, type='bool'), vol_nearly_full_threshold_percent=dict(required=False, type='int'), vol_full_threshold_percent=dict(required=False, type='int'), auto_provision_as=dict(choices=['flexgroup'], required=False, type='str'), wait_for_completion=dict(required=False, type='bool', default=False), time_out=dict(required=False, type='int', default=180), max_wait_time=dict(required=False, type='int', default=600), language=dict(type='str', required=False), qos_policy_group=dict(required=False, type='str'), qos_adaptive_policy_group=dict(required=False, type='str'), nvfail_enabled=dict(type='bool', required=False), space_slo=dict(type='str', required=False, choices=['none', 'thick', 'semi-thick']), tiering_policy=dict(type='str', required=False, choices=['snapshot-only', 'auto', 'backup', 'none', 'all']), vserver_dr_protection=dict(type='str', required=False, choices=['protected', 'unprotected']), comment=dict(type='str', required=False), snapshot_auto_delete=dict(type='dict', required=False), cutover_action=dict(required=False, type='str', choices=['abort_on_failure', 'defer_on_failure', 'force', 'wait']), check_interval=dict(required=False, type='int', default=30), from_vserver=dict(required=False, type='str'), auto_remap_luns=dict(required=False, type='bool'), force_unmap_luns=dict(required=False, type='bool'), force_restore=dict(required=False, type='bool'), compression=dict(required=False, type='bool'), inline_compression=dict(required=False, type='bool'), preserve_lun_ids=dict(required=False, type='bool'), snapshot_restore=dict(required=False, type='str'), nas_application_template=dict(type='dict', options=dict(use_nas_application=dict(type='bool', default=True), exclude_aggregates=dict(type='list', elements='str'), flexcache=dict(type='dict', options=dict(dr_cache=dict(type='bool'), origin_svm_name=dict(required=True, type='str'), origin_component_name=dict(required=True, type='str'))), cifs_access=dict(type='list', elements='dict', options=dict(access=dict(type='str', choices=['change', 'full_control', 'no_access', 'read']), user_or_group=dict(type='str'))), nfs_access=dict(type='list', elements='dict', options=dict(access=dict(type='str', choices=['none', 'ro', 'rw']), host=dict(type='str'))), storage_service=dict(type='str', choices=['value', 'performance', 'extreme']), tiering=dict(type='dict', options=dict(control=dict(type='str', choices=['required', 'best_effort', 'disallowed']), policy=dict(type='str', choices=['all', 'auto', 'none', 'snapshot-only']), object_stores=dict(type='list', elements='str'))))), size_change_threshold=dict(type='int', default=10), tiering_minimum_cooling_days=dict(required=False, type='int'), logical_space_enforcement=dict(required=False, type='bool'), logical_space_reporting=dict(required=False, type='bool'), snaplock=dict(type='dict', options=dict(append_mode_enabled=dict(required=False, type='bool'), autocommit_period=dict(required=False, type='str'), privileged_delete=dict(required=False, type='str', choices=['disabled', 'enabled', 'permanently_disabled']), retention=dict(type='dict', options=dict(default=dict(required=False, type='str'), maximum=dict(required=False, type='str'), minimum=dict(required=False, type='str'))), type=dict(required=False, type='str', choices=['compliance', 'enterprise', 'non_snaplock']))), max_files=dict(required=False, type='int'), analytics=dict(required=False, type='str', choices=['on', 'off']), tags=dict(required=False, type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, mutually_exclusive=[['space_guarantee', 'space_slo'], ['auto_remap_luns', 'force_unmap_luns']], supports_check_mode=True)
        self.na_helper = NetAppModule(self)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.volume_style = None
        self.volume_created = False
        self.issues = []
        self.sis_keys2zapi_get = dict(efficiency_policy='policy', compression='is-compression-enabled', inline_compression='is-inline-compression-enabled')
        self.sis_keys2zapi_set = dict(efficiency_policy='policy-name', compression='enable-compression', inline_compression='enable-inline-compression')
        if self.parameters.get('size'):
            self.parameters['size'] = self.parameters['size'] * netapp_utils.POW2_BYTE_MAP[self.parameters['size_unit']]
        self.validate_snapshot_auto_delete()
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        unsupported_rest_properties = ['cutover_action', 'encrypt-destination', 'force_restore', 'nvfail_enabled', 'preserve_lun_ids', 'destroy_list', 'space_slo', 'vserver_dr_protection']
        partially_supported_rest_properties = [['efficiency_policy', (9, 7)], ['tiering_minimum_cooling_days', (9, 8)], ['analytics', (9, 8)], ['atime_update', (9, 8)], ['vol_nearly_full_threshold_percent', (9, 9)], ['vol_full_threshold_percent', (9, 9)], ['tags', (9, 13, 1)], ['snapdir_access', (9, 13, 1)], ['snapshot_auto_delete', (9, 13, 1)]]
        self.unsupported_zapi_properties = ['sizing_method', 'logical_space_enforcement', 'logical_space_reporting', 'snaplock', 'analytics', 'tags', 'vol_nearly_full_threshold_percent', 'vol_full_threshold_percent']
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if not self.use_rest:
            self.setup_zapi()
        if self.use_rest:
            self.rest_errors()
        self.rest_app = self.setup_rest_application()

    def setup_zapi(self):
        if netapp_utils.has_netapp_lib() is False:
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        for unsupported_zapi_property in self.unsupported_zapi_properties:
            if self.parameters.get(unsupported_zapi_property) is not None:
                msg = 'Error: %s option is not supported with ZAPI.  It can only be used with REST.' % unsupported_zapi_property
                msg += '  use_rest: %s.' % self.parameters['use_rest']
                if self.rest_api.fallback_to_zapi_reason:
                    msg += '  Conflict %s.' % self.rest_api.fallback_to_zapi_reason
                self.module.fail_json(msg=msg)
        self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
        self.cluster = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def validate_snapshot_auto_delete(self):
        if 'snapshot_auto_delete' in self.parameters:
            for key in self.parameters['snapshot_auto_delete']:
                if key not in ['commitment', 'trigger', 'target_free_space', 'delete_order', 'defer_delete', 'prefix', 'destroy_list', 'state']:
                    self.module.fail_json(msg="snapshot_auto_delete option '%s' is not valid." % key)

    def setup_rest_application(self):
        rest_app = None
        if self.na_helper.safe_get(self.parameters, ['nas_application_template', 'use_nas_application']):
            if not self.use_rest:
                msg = 'Error: nas_application_template requires REST support.'
                msg += '  use_rest: %s.' % self.parameters['use_rest']
                if self.rest_api.fallback_to_zapi_reason:
                    msg += '  Conflict %s.' % self.rest_api.fallback_to_zapi_reason
                self.module.fail_json(msg=msg)
            tiering_policy_nas = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'tiering', 'policy'])
            tiering_policy = self.na_helper.safe_get(self.parameters, ['tiering_policy'])
            if tiering_policy_nas is not None and tiering_policy is not None and (tiering_policy_nas != tiering_policy):
                msg = 'Conflict: if tiering_policy and nas_application_template tiering policy are both set, they must match.'
                msg += '  Found "%s" and "%s".' % (tiering_policy, tiering_policy_nas)
                self.module.fail_json(msg=msg)
            if self.parameters.get('aggregate_name') is not None:
                msg = 'Conflict: aggregate_name is not supported when application template is enabled.  Found: aggregate_name: %s' % self.parameters['aggregate_name']
                self.module.fail_json(msg=msg)
            nfs_access = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'nfs_access'])
            if nfs_access is not None and self.na_helper.safe_get(self.parameters, ['export_policy']) is not None:
                msg = 'Conflict: export_policy option and nfs_access suboption in nas_application_template are mutually exclusive.'
                self.module.fail_json(msg=msg)
            rest_app = RestApplication(self.rest_api, self.parameters['vserver'], self.parameters['name'])
        return rest_app

    def volume_get_iter(self, vol_name=None):
        """
        Return volume-get-iter query results
        :param vol_name: name of the volume
        :return: NaElement
        """
        volume_info = netapp_utils.zapi.NaElement('volume-get-iter')
        volume_attributes = netapp_utils.zapi.NaElement('volume-attributes')
        volume_id_attributes = netapp_utils.zapi.NaElement('volume-id-attributes')
        volume_id_attributes.add_new_child('name', vol_name)
        volume_id_attributes.add_new_child('vserver', self.parameters['vserver'])
        volume_attributes.add_child_elem(volume_id_attributes)
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(volume_attributes)
        volume_info.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(volume_info, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching volume %s : %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return result

    def get_application(self):
        if self.rest_app:
            app, error = self.rest_app.get_application_details('nas')
            self.na_helper.fail_on_error(error)
            comps = self.na_helper.safe_get(app, ['nas', 'application_components'])
            if comps:
                comp = comps[0]
                app['nas'].pop('application_components')
                app['nas'].update(comp)
                return app['nas']
        return None

    def get_volume_attributes(self, volume_attributes, result):
        attrs = dict(encrypt=dict(key_list=['encrypt'], convert_to=bool, omitnone=True), tiering_policy=dict(key_list=['volume-comp-aggr-attributes', 'tiering-policy'], omitnone=True), export_policy=dict(key_list=['volume-export-attributes', 'policy']), aggregate_name=dict(key_list=['volume-id-attributes', 'containing-aggregate-name']), flexgroup_uuid=dict(key_list=['volume-id-attributes', 'flexgroup-uuid']), instance_uuid=dict(key_list=['volume-id-attributes', 'instance-uuid']), junction_path=dict(key_list=['volume-id-attributes', 'junction-path'], default=''), style_extended=dict(key_list=['volume-id-attributes', 'style-extended']), type=dict(key_list=['volume-id-attributes', 'type'], omitnone=True), comment=dict(key_list=['volume-id-attributes', 'comment']), max_files=dict(key_list=['volume-inode-attributes', 'files-total'], convert_to=int), atime_update=dict(key_list=['volume-performance-attributes', 'is-atime-update-enabled'], convert_to=bool), qos_policy_group=dict(key_list=['volume-qos-attributes', 'policy-group-name']), qos_adaptive_policy_group=dict(key_list=['volume-qos-attributes', 'adaptive-policy-group-name']), volume_security_style=dict(key_list=['volume-security-attributes', 'style'], omitnone=True), group_id=dict(key_list=['volume-security-attributes', 'volume-security-unix-attributes', 'group-id'], convert_to=int, omitnone=True), unix_permissions=dict(key_list=['volume-security-attributes', 'volume-security-unix-attributes', 'permissions'], required=True), user_id=dict(key_list=['volume-security-attributes', 'volume-security-unix-attributes', 'user-id'], convert_to=int, omitnone=True), snapdir_access=dict(key_list=['volume-snapshot-attributes', 'snapdir-access-enabled'], convert_to=bool), snapshot_policy=dict(key_list=['volume-snapshot-attributes', 'snapshot-policy'], omitnone=True), percent_snapshot_space=dict(key_list=['volume-space-attributes', 'percentage-snapshot-reserve'], convert_to=int, omitnone=True), size=dict(key_list=['volume-space-attributes', 'size'], convert_to=int), space_guarantee=dict(key_list=['volume-space-attributes', 'space-guarantee']), space_slo=dict(key_list=['volume-space-attributes', 'space-slo']), nvfail_enabled=dict(key_list=['volume-state-attributes', 'is-nvfail-enabled'], convert_to=bool), is_online=dict(key_list=['volume-state-attributes', 'state'], convert_to='bool_online', omitnone=True), vserver_dr_protection=dict(key_list=['volume-vserver-dr-protection-attributes', 'vserver-dr-protection']))
        self.na_helper.zapi_get_attrs(volume_attributes, attrs, result)

    def get_snapshot_auto_delete_attributes(self, volume_attributes, result):
        attrs = dict(commitment=dict(key_list=['volume-snapshot-autodelete-attributes', 'commitment']), defer_delete=dict(key_list=['volume-snapshot-autodelete-attributes', 'defer-delete']), delete_order=dict(key_list=['volume-snapshot-autodelete-attributes', 'delete-order']), destroy_list=dict(key_list=['volume-snapshot-autodelete-attributes', 'destroy-list']), is_autodelete_enabled=dict(key_list=['volume-snapshot-autodelete-attributes', 'is-autodelete-enabled'], convert_to=bool), prefix=dict(key_list=['volume-snapshot-autodelete-attributes', 'prefix']), target_free_space=dict(key_list=['volume-snapshot-autodelete-attributes', 'target-free-space'], convert_to=int), trigger=dict(key_list=['volume-snapshot-autodelete-attributes', 'trigger']))
        self.na_helper.zapi_get_attrs(volume_attributes, attrs, result)
        if result['is_autodelete_enabled'] is not None:
            result['state'] = 'on' if result['is_autodelete_enabled'] else 'off'
            del result['is_autodelete_enabled']

    def get_volume(self, vol_name=None):
        """
        Return details about the volume
        :param:
            name : Name of the volume
        :return: Details about the volume. None if not found.
        :rtype: dict
        """
        result = None
        if vol_name is None:
            vol_name = self.parameters['name']
        if self.use_rest:
            return self.get_volume_rest(vol_name)
        volume_info = self.volume_get_iter(vol_name)
        if self.na_helper.zapi_get_value(volume_info, ['num-records'], convert_to=int, default=0) > 0:
            result = self.get_volume_record_from_zapi(volume_info, vol_name)
        return result

    def get_volume_record_from_zapi(self, volume_info, vol_name):
        volume_attributes = self.na_helper.zapi_get_value(volume_info, ['attributes-list', 'volume-attributes'], required=True)
        result = dict(name=vol_name)
        self.get_volume_attributes(volume_attributes, result)
        result['uuid'] = result['instance_uuid'] if result['style_extended'] == 'flexvol' else result['flexgroup_uuid'] if result['style_extended'] is not None and result['style_extended'].startswith('flexgroup') else None
        auto_delete = {}
        self.get_snapshot_auto_delete_attributes(volume_attributes, auto_delete)
        result['snapshot_auto_delete'] = auto_delete
        self.get_efficiency_info(result)
        return result

    def wrap_fail_json(self, msg, exception=None):
        for issue in self.issues:
            self.module.warn(issue)
        if self.volume_created:
            msg = 'Volume created with success, with missing attributes: %s' % msg
        self.module.fail_json(msg=msg, exception=exception)

    def create_nas_application_component(self):
        """Create application component for nas template"""
        required_options = ('name', 'size')
        for option in required_options:
            if self.parameters.get(option) is None:
                self.module.fail_json(msg='Error: "%s" is required to create nas application.' % option)
        application_component = dict(name=self.parameters['name'], total_size=self.parameters['size'], share_count=1, scale_out=self.volume_style == 'flexgroup')
        name = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'storage_service'])
        if name is not None:
            application_component['storage_service'] = dict(name=name)
        flexcache = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache'])
        if flexcache is not None:
            application_component['flexcache'] = dict(origin=dict(svm=dict(name=flexcache['origin_svm_name']), component=dict(name=flexcache['origin_component_name'])))
            del application_component['scale_out']
            dr_cache = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache', 'dr_cache'])
            if dr_cache is not None:
                application_component['flexcache']['dr_cache'] = dr_cache
        tiering = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'tiering'])
        if tiering is not None or self.parameters.get('tiering_policy') is not None:
            application_component['tiering'] = {}
            if tiering is None:
                tiering = {}
            if 'policy' not in tiering:
                tiering['policy'] = self.parameters.get('tiering_policy')
            for attr in ('control', 'policy', 'object_stores'):
                value = tiering.get(attr)
                if attr == 'object_stores' and value is not None:
                    value = [dict(name=x) for x in value]
                if value is not None:
                    application_component['tiering'][attr] = value
        if self.get_qos_policy_group() is not None:
            application_component['qos'] = {'policy': {'name': self.get_qos_policy_group()}}
        if self.parameters.get('export_policy') is not None:
            application_component['export_policy'] = {'name': self.parameters['export_policy']}
        return application_component

    def create_volume_body(self):
        """Create body for nas template"""
        nas = dict(application_components=[self.create_nas_application_component()])
        value = self.na_helper.safe_get(self.parameters, ['snapshot_policy'])
        if value is not None:
            nas['protection_type'] = {'local_policy': value}
        for attr in ('nfs_access', 'cifs_access'):
            value = self.na_helper.safe_get(self.parameters, ['nas_application_template', attr])
            if value is not None:
                value = self.na_helper.filter_out_none_entries(value)
                if value:
                    nas[attr] = value
        for attr in ('exclude_aggregates',):
            values = self.na_helper.safe_get(self.parameters, ['nas_application_template', attr])
            if values:
                nas[attr] = [dict(name=name) for name in values]
        return self.rest_app.create_application_body('nas', nas, smart_container=True)

    def create_nas_application(self):
        """Use REST application/applications nas template to create a volume"""
        body, error = self.create_volume_body()
        self.na_helper.fail_on_error(error)
        response, error = self.rest_app.create_application(body)
        self.na_helper.fail_on_error(error)
        return response

    def wait_for_volume_online(self, sleep_time=10):
        retries = (self.parameters['time_out'] + 5) // 10
        is_online = None
        errors = []
        while not is_online and retries > 0:
            try:
                current = self.get_volume()
                is_online = None if current is None else current['is_online']
            except KeyError as err:
                errors.append(repr(err))
            if not is_online:
                time.sleep(sleep_time)
            retries -= 1
        if not is_online:
            errors.append('Timeout after %s seconds' % self.parameters['time_out'])
            self.module.fail_json(msg='Error waiting for volume %s to come online: %s' % (self.parameters['name'], str(errors)))

    def create_volume(self):
        """Create ONTAP volume"""
        if self.rest_app:
            return self.create_nas_application()
        if self.use_rest:
            return self.create_volume_rest()
        if self.volume_style == 'flexgroup':
            return self.create_volume_async()
        options = self.create_volume_options()
        volume_create = netapp_utils.zapi.NaElement.create_node_with_children('volume-create', **options)
        try:
            self.server.invoke_successfully(volume_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            size_msg = ' of size %s' % self.parameters['size'] if self.parameters.get('size') is not None else ''
            self.module.fail_json(msg='Error provisioning volume %s%s: %s' % (self.parameters['name'], size_msg, to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_online()
        return None

    def create_volume_async(self):
        """
        create volume async.
        """
        options = self.create_volume_options()
        volume_create = netapp_utils.zapi.NaElement.create_node_with_children('volume-create-async', **options)
        if self.parameters.get('aggr_list'):
            aggr_list_obj = netapp_utils.zapi.NaElement('aggr-list')
            volume_create.add_child_elem(aggr_list_obj)
            for aggr in self.parameters['aggr_list']:
                aggr_list_obj.add_new_child('aggr-name', aggr)
        try:
            result = self.server.invoke_successfully(volume_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            size_msg = ' of size %s' % self.parameters['size'] if self.parameters.get('size') is not None else ''
            self.module.fail_json(msg='Error provisioning volume %s%s: %s' % (self.parameters['name'], size_msg, to_native(error)), exception=traceback.format_exc())
        self.check_invoke_result(result, 'create')
        return None

    def create_volume_options(self):
        """Set volume options for create operation"""
        options = {}
        if self.volume_style == 'flexgroup':
            options['volume-name'] = self.parameters['name']
            if self.parameters.get('aggr_list_multiplier') is not None:
                options['aggr-list-multiplier'] = str(self.parameters['aggr_list_multiplier'])
            if self.parameters.get('auto_provision_as') is not None:
                options['auto-provision-as'] = self.parameters['auto_provision_as']
            if self.parameters.get('space_guarantee') is not None:
                options['space-guarantee'] = self.parameters['space_guarantee']
        else:
            options['volume'] = self.parameters['name']
            if self.parameters.get('aggregate_name') is None:
                self.module.fail_json(msg='Error provisioning volume %s: aggregate_name is required' % self.parameters['name'])
            options['containing-aggr-name'] = self.parameters['aggregate_name']
            if self.parameters.get('space_guarantee') is not None:
                options['space-reserve'] = self.parameters['space_guarantee']
        if self.parameters.get('size') is not None:
            options['size'] = str(self.parameters['size'])
        if self.parameters.get('snapshot_policy') is not None:
            options['snapshot-policy'] = self.parameters['snapshot_policy']
        if self.parameters.get('unix_permissions') is not None:
            options['unix-permissions'] = self.parameters['unix_permissions']
        if self.parameters.get('group_id') is not None:
            options['group-id'] = str(self.parameters['group_id'])
        if self.parameters.get('user_id') is not None:
            options['user-id'] = str(self.parameters['user_id'])
        if self.parameters.get('volume_security_style') is not None:
            options['volume-security-style'] = self.parameters['volume_security_style']
        if self.parameters.get('export_policy') is not None:
            options['export-policy'] = self.parameters['export_policy']
        if self.parameters.get('junction_path') is not None:
            options['junction-path'] = self.parameters['junction_path']
        if self.parameters.get('comment') is not None:
            options['volume-comment'] = self.parameters['comment']
        if self.parameters.get('type') is not None:
            options['volume-type'] = self.parameters['type']
        if self.parameters.get('percent_snapshot_space') is not None:
            options['percentage-snapshot-reserve'] = str(self.parameters['percent_snapshot_space'])
        if self.parameters.get('language') is not None:
            options['language-code'] = self.parameters['language']
        if self.parameters.get('qos_policy_group') is not None:
            options['qos-policy-group-name'] = self.parameters['qos_policy_group']
        if self.parameters.get('qos_adaptive_policy_group') is not None:
            options['qos-adaptive-policy-group-name'] = self.parameters['qos_adaptive_policy_group']
        if self.parameters.get('nvfail_enabled') is not None:
            options['is-nvfail-enabled'] = str(self.parameters['nvfail_enabled'])
        if self.parameters.get('space_slo') is not None:
            options['space-slo'] = self.parameters['space_slo']
        if self.parameters.get('tiering_policy') is not None:
            options['tiering-policy'] = self.parameters['tiering_policy']
        if self.parameters.get('encrypt') is not None:
            options['encrypt'] = self.na_helper.get_value_for_bool(False, self.parameters['encrypt'], 'encrypt')
        if self.parameters.get('vserver_dr_protection') is not None:
            options['vserver-dr-protection'] = self.parameters['vserver_dr_protection']
        if self.parameters['is_online']:
            options['volume-state'] = 'online'
        else:
            options['volume-state'] = 'offline'
        return options

    def rest_delete_volume(self, current):
        """
        Delete the volume using REST DELETE method (it scrubs better than ZAPI).
        """
        uuid = self.parameters['uuid']
        if uuid is None:
            self.module.fail_json(msg='Could not read UUID for volume %s in delete.' % self.parameters['name'])
        unmount_error = self.volume_unmount_rest(fail_on_error=False) if current.get('junction_path') else None
        dummy, error = rest_generic.delete_async(self.rest_api, 'storage/volumes', uuid, job_timeout=self.parameters['time_out'])
        self.na_helper.fail_on_error(error, previous_errors=['Error unmounting volume: %s' % unmount_error] if unmount_error else None)
        if unmount_error:
            self.module.warn('Volume was successfully deleted though unmount failed with: %s' % unmount_error)

    def delete_volume_async(self, current):
        """Delete ONTAP volume for infinite or flexgroup types """
        errors = None
        if current['is_online']:
            dummy, errors = self.change_volume_state(call_from_delete_vol=True)
        volume_delete = netapp_utils.zapi.NaElement.create_node_with_children('volume-destroy-async', **{'volume-name': self.parameters['name']})
        try:
            result = self.server.invoke_successfully(volume_delete, enable_tunneling=True)
            self.check_invoke_result(result, 'delete')
        except netapp_utils.zapi.NaApiError as error:
            msg = 'Error deleting volume %s: %s.' % (self.parameters['name'], to_native(error))
            if errors:
                msg += '  Previous errors when offlining/unmounting volume: %s' % ' - '.join(errors)
            self.module.fail_json(msg=msg)

    def delete_volume_sync(self, current, unmount_offline):
        """Delete ONTAP volume for flexvol types """
        options = {'name': self.parameters['name']}
        if unmount_offline:
            options['unmount-and-offline'] = 'true'
        volume_delete = netapp_utils.zapi.NaElement.create_node_with_children('volume-destroy', **options)
        try:
            self.server.invoke_successfully(volume_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            return error
        return None

    def delete_volume(self, current):
        """Delete ONTAP volume"""
        if self.use_rest and self.parameters['uuid'] is not None:
            return self.rest_delete_volume(current)
        if self.parameters.get('is_infinite') or self.volume_style == 'flexgroup':
            return self.delete_volume_async(current)
        errors = []
        error = self.delete_volume_sync(current, True)
        if error:
            errors.append('volume delete failed with unmount-and-offline option: %s' % to_native(error))
            error = self.delete_volume_sync(current, False)
        if error:
            errors.append('volume delete failed without unmount-and-offline option: %s' % to_native(error))
        if errors:
            self.module.fail_json(msg='Error deleting volume %s: %s' % (self.parameters['name'], ' - '.join(errors)), exception=traceback.format_exc())

    def move_volume(self, encrypt_destination=None):
        """Move volume from source aggregate to destination aggregate"""
        if self.use_rest:
            return self.move_volume_rest(encrypt_destination)
        volume_move = netapp_utils.zapi.NaElement.create_node_with_children('volume-move-start', **{'source-volume': self.parameters['name'], 'vserver': self.parameters['vserver'], 'dest-aggr': self.parameters['aggregate_name']})
        if self.parameters.get('cutover_action'):
            volume_move.add_new_child('cutover-action', self.parameters['cutover_action'])
        if encrypt_destination is not None:
            volume_move.add_new_child('encrypt-destination', self.na_helper.get_value_for_bool(False, encrypt_destination))
        try:
            self.cluster.invoke_successfully(volume_move, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            rest_error = self.move_volume_with_rest_passthrough(encrypt_destination)
            if rest_error is not None:
                self.module.fail_json(msg='Error moving volume %s: %s -  Retry failed with REST error: %s' % (self.parameters['name'], to_native(error), rest_error), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_move()

    def move_volume_with_rest_passthrough(self, encrypt_destination=None):
        if not self.use_rest:
            return False
        api = 'private/cli/volume/move/start'
        body = {'destination-aggregate': self.parameters['aggregate_name']}
        if encrypt_destination is not None:
            body['encrypt-destination'] = encrypt_destination
        query = {'volume': self.parameters['name'], 'vserver': self.parameters['vserver']}
        dummy, error = self.rest_api.patch(api, body, query)
        return error

    def check_volume_move_state(self, result):
        if self.use_rest:
            volume_move_status = self.na_helper.safe_get(result, ['movement', 'state'])
        else:
            volume_move_status = result.get_child_by_name('attributes-list').get_child_by_name('volume-move-info').get_child_content('state')
        if volume_move_status in ['success', 'done']:
            return False
        if volume_move_status in ['failed', 'alert', 'aborted']:
            self.module.fail_json(msg='Error moving volume %s: %s' % (self.parameters['name'], result.get_child_by_name('attributes-list').get_child_by_name('volume-move-info').get_child_content('details')))
        return True

    def wait_for_volume_move(self):
        volume_move_iter = netapp_utils.zapi.NaElement('volume-move-get-iter')
        volume_move_info = netapp_utils.zapi.NaElement('volume-move-info')
        volume_move_info.add_new_child('volume', self.parameters['name'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(volume_move_info)
        volume_move_iter.add_child_elem(query)
        error = self.wait_for_task_completion(volume_move_iter, self.check_volume_move_state)
        if error:
            self.module.fail_json(msg='Error getting volume move status: %s' % to_native(error), exception=traceback.format_exc())

    def wait_for_volume_move_rest(self):
        api = 'storage/volumes'
        query = {'name': self.parameters['name'], 'movement.destination_aggregate.name': self.parameters['aggregate_name'], 'fields': 'movement.state'}
        error = self.wait_for_task_completion_rest(api, query, self.check_volume_move_state)
        if error:
            self.module.fail_json(msg='Error getting volume move status: %s' % to_native(error), exception=traceback.format_exc())

    def check_volume_encryption_conversion_state(self, result):
        if self.use_rest:
            volume_encryption_conversion_status = self.na_helper.safe_get(result, ['encryption', 'status', 'message'])
        else:
            volume_encryption_conversion_status = result.get_child_by_name('attributes-list').get_child_by_name('volume-encryption-conversion-info').get_child_content('status')
        if volume_encryption_conversion_status in ['running', 'initializing']:
            return True
        if volume_encryption_conversion_status in ['Not currently going on.', None]:
            return False
        self.module.fail_json(msg='Error converting encryption for volume %s: %s' % (self.parameters['name'], volume_encryption_conversion_status))

    def wait_for_volume_encryption_conversion(self):
        if self.use_rest:
            return self.wait_for_volume_encryption_conversion_rest()
        volume_encryption_conversion_iter = netapp_utils.zapi.NaElement('volume-encryption-conversion-get-iter')
        volume_encryption_conversion_info = netapp_utils.zapi.NaElement('volume-encryption-conversion-info')
        volume_encryption_conversion_info.add_new_child('volume', self.parameters['name'])
        volume_encryption_conversion_info.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(volume_encryption_conversion_info)
        volume_encryption_conversion_iter.add_child_elem(query)
        error = self.wait_for_task_completion(volume_encryption_conversion_iter, self.check_volume_encryption_conversion_state)
        if error:
            self.module.fail_json(msg='Error getting volume encryption_conversion status: %s' % to_native(error), exception=traceback.format_exc())

    def wait_for_volume_encryption_conversion_rest(self):
        api = 'storage/volumes'
        query = {'name': self.parameters['name'], 'fields': 'encryption'}
        error = self.wait_for_task_completion_rest(api, query, self.check_volume_encryption_conversion_state)
        if error:
            self.module.fail_json(msg='Error getting volume encryption_conversion status: %s' % to_native(error), exception=traceback.format_exc())

    def wait_for_task_completion(self, zapi_iter, check_state):
        retries = self.parameters['max_wait_time'] // (self.parameters['check_interval'] + 1)
        fail_count = 0
        while retries > 0:
            try:
                result = self.cluster.invoke_successfully(zapi_iter, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                if fail_count < 3:
                    fail_count += 1
                    retries -= 1
                    time.sleep(self.parameters['check_interval'])
                    continue
                return error
            if int(result.get_child_content('num-records')) == 0:
                return None
            fail_count = 0
            retry_required = check_state(result)
            if not retry_required:
                return None
            time.sleep(self.parameters['check_interval'])
            retries -= 1

    def wait_for_task_completion_rest(self, api, query, check_state):
        retries = self.parameters['max_wait_time'] // (self.parameters['check_interval'] + 1)
        fail_count = 0
        while retries > 0:
            record, error = rest_generic.get_one_record(self.rest_api, api, query)
            if error:
                if fail_count < 3:
                    fail_count += 1
                    retries -= 1
                    time.sleep(self.parameters['check_interval'])
                    continue
                return error
            if record is None:
                return None
            fail_count = 0
            retry_required = check_state(record)
            if not retry_required:
                return None
            time.sleep(self.parameters['check_interval'])
            retries -= 1

    def rename_volume(self):
        """
        Rename the volume.

        Note: 'is_infinite' needs to be set to True in order to rename an
        Infinite Volume. Use time_out parameter to set wait time for rename completion.
        """
        if self.use_rest:
            return self.rename_volume_rest()
        vol_rename_zapi, vol_name_zapi = ['volume-rename-async', 'volume-name'] if self.parameters['is_infinite'] else ['volume-rename', 'volume']
        volume_rename = netapp_utils.zapi.NaElement.create_node_with_children(vol_rename_zapi, **{vol_name_zapi: self.parameters['from_name'], 'new-volume-name': str(self.parameters['name'])})
        try:
            result = self.server.invoke_successfully(volume_rename, enable_tunneling=True)
            if vol_rename_zapi == 'volume-rename-async':
                self.check_invoke_result(result, 'rename')
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def resize_volume(self):
        """
        Re-size the volume.

        Note: 'is_infinite' needs to be set to True in order to resize an
        Infinite Volume.
        """
        if self.use_rest:
            return self.resize_volume_rest()
        vol_size_zapi, vol_name_zapi = ['volume-size-async', 'volume-name'] if self.parameters['is_infinite'] or self.volume_style == 'flexgroup' else ['volume-size', 'volume']
        volume_resize = netapp_utils.zapi.NaElement.create_node_with_children(vol_size_zapi, **{vol_name_zapi: self.parameters['name'], 'new-size': str(self.parameters['size'])})
        try:
            result = self.server.invoke_successfully(volume_resize, enable_tunneling=True)
            if vol_size_zapi == 'volume-size-async':
                self.check_invoke_result(result, 'resize')
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error re-sizing volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return None

    def start_encryption_conversion(self, encrypt_destination):
        if encrypt_destination:
            if self.use_rest:
                return self.encryption_conversion_rest()
            zapi = netapp_utils.zapi.NaElement.create_node_with_children('volume-encryption-conversion-start', **{'volume': self.parameters['name']})
            try:
                self.server.invoke_successfully(zapi, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error enabling encryption for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
            if self.parameters.get('wait_for_completion'):
                self.wait_for_volume_encryption_conversion()
        else:
            self.module.warn('disabling encryption requires cluster admin permissions.')
            self.move_volume(encrypt_destination)

    def change_volume_state(self, call_from_delete_vol=False):
        """
        Change volume's state (offline/online).
        """
        if self.use_rest:
            return self.change_volume_state_rest()
        if self.parameters['is_online'] and (not call_from_delete_vol):
            vol_state_zapi, vol_name_zapi, action = ['volume-online-async', 'volume-name', 'online'] if self.parameters['is_infinite'] or self.volume_style == 'flexgroup' else ['volume-online', 'name', 'online']
        else:
            vol_state_zapi, vol_name_zapi, action = ['volume-offline-async', 'volume-name', 'offline'] if self.parameters['is_infinite'] or self.volume_style == 'flexgroup' else ['volume-offline', 'name', 'offline']
            volume_unmount = netapp_utils.zapi.NaElement.create_node_with_children('volume-unmount', **{'volume-name': self.parameters['name']})
        volume_change_state = netapp_utils.zapi.NaElement.create_node_with_children(vol_state_zapi, **{vol_name_zapi: self.parameters['name']})
        errors = []
        if not self.parameters['is_online'] or call_from_delete_vol:
            try:
                self.server.invoke_successfully(volume_unmount, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                errors.append('Error unmounting volume %s: %s' % (self.parameters['name'], to_native(error)))
        state = 'online' if self.parameters['is_online'] and (not call_from_delete_vol) else 'offline'
        try:
            result = self.server.invoke_successfully(volume_change_state, enable_tunneling=True)
            if self.volume_style == 'flexgroup' or self.parameters['is_infinite']:
                self.check_invoke_result(result, action)
        except netapp_utils.zapi.NaApiError as error:
            errors.append('Error changing the state of volume %s to %s: %s' % (self.parameters['name'], state, to_native(error)))
        if errors and (not call_from_delete_vol):
            self.module.fail_json(msg=', '.join(errors), exception=traceback.format_exc())
        return (state, errors)

    def create_volume_attribute(self, zapi_object, parent_attribute, attribute, option_name, convert_from=None):
        """

        :param parent_attribute:
        :param child_attribute:
        :param value:
        :return:
        """
        value = self.parameters.get(option_name)
        if value is None:
            return
        if convert_from == int:
            value = str(value)
        elif convert_from == bool:
            value = self.na_helper.get_value_for_bool(False, value, option_name)
        if zapi_object is None:
            parent_attribute.add_new_child(attribute, value)
            return
        if isinstance(zapi_object, str):
            element = parent_attribute.get_child_by_name(zapi_object)
            zapi_object = netapp_utils.zapi.NaElement(zapi_object) if element is None else element
        zapi_object.add_new_child(attribute, value)
        parent_attribute.add_child_elem(zapi_object)

    def build_zapi_volume_modify_iter(self, params):
        vol_mod_iter = netapp_utils.zapi.NaElement('volume-modify-iter-async' if self.volume_style == 'flexgroup' or self.parameters['is_infinite'] else 'volume-modify-iter')
        attributes = netapp_utils.zapi.NaElement('attributes')
        vol_mod_attributes = netapp_utils.zapi.NaElement('volume-attributes')
        vol_inode_attributes = netapp_utils.zapi.NaElement('volume-inode-attributes')
        self.create_volume_attribute(vol_inode_attributes, vol_mod_attributes, 'files-total', 'max_files', int)
        vol_space_attributes = netapp_utils.zapi.NaElement('volume-space-attributes')
        self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'space-guarantee', 'space_guarantee')
        self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'percentage-snapshot-reserve', 'percent_snapshot_space', int)
        self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'space-slo', 'space_slo')
        vol_snapshot_attributes = netapp_utils.zapi.NaElement('volume-snapshot-attributes')
        self.create_volume_attribute(vol_snapshot_attributes, vol_mod_attributes, 'snapshot-policy', 'snapshot_policy')
        self.create_volume_attribute(vol_snapshot_attributes, vol_mod_attributes, 'snapdir-access-enabled', 'snapdir_access', bool)
        self.create_volume_attribute('volume-export-attributes', vol_mod_attributes, 'policy', 'export_policy')
        if self.parameters.get('unix_permissions') is not None or self.parameters.get('group_id') is not None or self.parameters.get('user_id') is not None:
            vol_security_attributes = netapp_utils.zapi.NaElement('volume-security-attributes')
            vol_security_unix_attributes = netapp_utils.zapi.NaElement('volume-security-unix-attributes')
            self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'permissions', 'unix_permissions')
            self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'group-id', 'group_id', int)
            self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'user-id', 'user_id', int)
            vol_mod_attributes.add_child_elem(vol_security_attributes)
        if params and params.get('volume_security_style') is not None:
            self.create_volume_attribute('volume-security-attributes', vol_mod_attributes, 'style', 'volume_security_style')
        self.create_volume_attribute('volume-performance-attributes', vol_mod_attributes, 'is-atime-update-enabled', 'atime_update', bool)
        self.create_volume_attribute('volume-qos-attributes', vol_mod_attributes, 'policy-group-name', 'qos_policy_group')
        self.create_volume_attribute('volume-qos-attributes', vol_mod_attributes, 'adaptive-policy-group-name', 'qos_adaptive_policy_group')
        if params and params.get('tiering_policy') is not None:
            self.create_volume_attribute('volume-comp-aggr-attributes', vol_mod_attributes, 'tiering-policy', 'tiering_policy')
        self.create_volume_attribute('volume-state-attributes', vol_mod_attributes, 'is-nvfail-enabled', 'nvfail_enabled', bool)
        self.create_volume_attribute('volume-vserver-dr-protection-attributes', vol_mod_attributes, 'vserver-dr-protection', 'vserver_dr_protection')
        self.create_volume_attribute('volume-id-attributes', vol_mod_attributes, 'comment', 'comment')
        attributes.add_child_elem(vol_mod_attributes)
        query = netapp_utils.zapi.NaElement('query')
        vol_query_attributes = netapp_utils.zapi.NaElement('volume-attributes')
        self.create_volume_attribute('volume-id-attributes', vol_query_attributes, 'name', 'name')
        query.add_child_elem(vol_query_attributes)
        vol_mod_iter.add_child_elem(attributes)
        vol_mod_iter.add_child_elem(query)
        return vol_mod_iter

    def volume_modify_attributes(self, params):
        """
        modify volume parameter 'export_policy','unix_permissions','snapshot_policy','space_guarantee', 'percent_snapshot_space',
                                'qos_policy_group', 'qos_adaptive_policy_group'
        """
        if self.use_rest:
            return self.volume_modify_attributes_rest(params)
        vol_mod_iter = self.build_zapi_volume_modify_iter(params)
        try:
            result = self.server.invoke_successfully(vol_mod_iter, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            error_msg = to_native(error)
            if 'volume-comp-aggr-attributes' in error_msg:
                error_msg += '. Added info: tiering option requires 9.4 or later.'
            self.wrap_fail_json(msg='Error modifying volume %s: %s' % (self.parameters['name'], error_msg), exception=traceback.format_exc())
        failures = result.get_child_by_name('failure-list')
        if failures is not None:
            error_msgs = [failures.get_child_by_name(return_info).get_child_content('error-message') for return_info in ('volume-modify-iter-info', 'volume-modify-iter-async-info') if failures.get_child_by_name(return_info) is not None]
            if error_msgs and any((x is not None for x in error_msgs)):
                self.wrap_fail_json(msg='Error modifying volume %s: %s' % (self.parameters['name'], ' --- '.join(error_msgs)), exception=traceback.format_exc())
        if self.volume_style == 'flexgroup' or self.parameters['is_infinite']:
            success = self.na_helper.safe_get(result, ['success-list', 'volume-modify-iter-async-info'])
            results = {}
            for key in ('status', 'jobid'):
                if success and success.get_child_by_name(key):
                    results[key] = success[key]
            status = results.get('status')
            if status == 'in_progress' and 'jobid' in results:
                if self.parameters['time_out'] == 0:
                    return
                error = self.check_job_status(results['jobid'])
                if error is None:
                    return
                self.wrap_fail_json(msg='Error when modifying volume: %s' % error)
            self.wrap_fail_json(msg='Unexpected error when modifying volume: result is: %s' % str(result.to_string()))

    def volume_mount(self):
        """
        Mount an existing volume in specified junction_path
        :return: None
        """
        if self.use_rest:
            return self.volume_mount_rest()
        vol_mount = netapp_utils.zapi.NaElement('volume-mount')
        vol_mount.add_new_child('volume-name', self.parameters['name'])
        vol_mount.add_new_child('junction-path', self.parameters['junction_path'])
        try:
            self.server.invoke_successfully(vol_mount, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error mounting volume %s on path %s: %s' % (self.parameters['name'], self.parameters['junction_path'], to_native(error)), exception=traceback.format_exc())

    def volume_unmount(self):
        """
        Unmount an existing volume
        :return: None
        """
        if self.use_rest:
            return self.volume_unmount_rest()
        vol_unmount = netapp_utils.zapi.NaElement.create_node_with_children('volume-unmount', **{'volume-name': self.parameters['name']})
        try:
            self.server.invoke_successfully(vol_unmount, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error unmounting volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_volume(self, modify):
        """Modify volume action"""
        if modify.get('junction_path') == '':
            self.volume_unmount()
        attributes = modify.keys()
        for attribute in attributes:
            if attribute in ['space_guarantee', 'export_policy', 'unix_permissions', 'group_id', 'user_id', 'tiering_policy', 'snapshot_policy', 'percent_snapshot_space', 'snapdir_access', 'atime_update', 'volume_security_style', 'nvfail_enabled', 'space_slo', 'qos_policy_group', 'qos_adaptive_policy_group', 'vserver_dr_protection', 'comment', 'logical_space_enforcement', 'logical_space_reporting', 'tiering_minimum_cooling_days', 'snaplock', 'max_files', 'analytics', 'tags', 'snapshot_auto_delete', 'vol_nearly_full_threshold_percent', 'vol_full_threshold_percent']:
                self.volume_modify_attributes(modify)
                break
        if 'snapshot_auto_delete' in attributes and (not self.use_rest):
            self.set_snapshot_auto_delete()
        if modify.get('junction_path'):
            self.volume_mount()
        if 'size' in attributes:
            self.resize_volume()
        if 'aggregate_name' in attributes:
            self.move_volume(modify.get('encrypt'))
        elif 'encrypt' in attributes:
            self.start_encryption_conversion(self.parameters['encrypt'])

    def get_volume_style(self, current):
        """Get volume style, infinite or standard flexvol"""
        if current is not None:
            return current.get('style_extended')
        if self.parameters.get('aggr_list') or self.parameters.get('aggr_list_multiplier') or self.parameters.get('auto_provision_as'):
            if self.use_rest and self.parameters.get('auto_provision_as') and (self.parameters.get('aggr_list_multiplier') is None):
                self.parameters['aggr_list_multiplier'] = 1
            return 'flexgroup'
        return None

    def get_job(self, jobid, server):
        """
        Get job details by id
        """
        job_get = netapp_utils.zapi.NaElement('job-get')
        job_get.add_new_child('job-id', jobid)
        try:
            result = server.invoke_successfully(job_get, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) == '15661':
                return None
            self.wrap_fail_json(msg='Error fetching job info: %s' % to_native(error), exception=traceback.format_exc())
        job_info = result.get_child_by_name('attributes').get_child_by_name('job-info')
        return {'job-progress': job_info['job-progress'], 'job-state': job_info['job-state'], 'job-completion': job_info['job-completion'] if job_info.get_child_by_name('job-completion') is not None else None}

    def check_job_status(self, jobid):
        """
        Loop until job is complete
        """
        server = self.server
        sleep_time = 5
        time_out = self.parameters['time_out']
        error = 'timeout'
        if time_out <= 0:
            results = self.get_job(jobid, server)
        while time_out > 0:
            results = self.get_job(jobid, server)
            if results is None and server == self.server:
                results = netapp_utils.get_cserver(self.server)
                server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=results)
                continue
            if results is None:
                error = 'cannot locate job with id: %d' % int(jobid)
                break
            if results['job-state'] in ('queued', 'running'):
                time.sleep(sleep_time)
                time_out -= sleep_time
                continue
            if results['job-state'] in ('success', 'failure'):
                break
            else:
                self.wrap_fail_json(msg='Unexpected job status in: %s' % repr(results))
        if results is not None:
            if results['job-state'] == 'success':
                error = None
            elif results['job-state'] in ('queued', 'running'):
                error = 'job completion exceeded expected timer of: %s seconds' % self.parameters['time_out']
            elif results['job-completion'] is not None:
                error = results['job-completion']
            else:
                error = results['job-progress']
        return error

    def check_invoke_result(self, result, action):
        """
        check invoked api call back result.
        """
        results = {}
        for key in ('result-status', 'result-jobid'):
            if result.get_child_by_name(key):
                results[key] = result[key]
        status = results.get('result-status')
        if status == 'in_progress' and 'result-jobid' in results:
            if self.parameters['time_out'] == 0:
                return
            error = self.check_job_status(results['result-jobid'])
            if error is None:
                return
            else:
                self.wrap_fail_json(msg='Error when %s volume: %s' % (action, error))
        if status == 'failed':
            self.wrap_fail_json(msg='Operation failed when %s volume.' % action)

    def set_efficiency_attributes(self, options):
        for key, attr in self.sis_keys2zapi_set.items():
            value = self.parameters.get(key)
            if value is not None:
                if self.argument_spec[key]['type'] == 'bool':
                    value = self.na_helper.get_value_for_bool(False, value)
                options[attr] = value
        if options.get('enable-inline-compression') == 'true' and 'enable-compression' not in options:
            options['enable-compression'] = 'true'

    def set_efficiency_config(self):
        """Set efficiency policy and compression attributes"""
        options = {'path': '/vol/' + self.parameters['name']}
        efficiency_enable = netapp_utils.zapi.NaElement.create_node_with_children('sis-enable', **options)
        try:
            self.server.invoke_successfully(efficiency_enable, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) != '40043':
                self.wrap_fail_json(msg='Error enable efficiency on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        self.set_efficiency_attributes(options)
        efficiency_start = netapp_utils.zapi.NaElement.create_node_with_children('sis-set-config', **options)
        try:
            self.server.invoke_successfully(efficiency_start, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.wrap_fail_json(msg='Error setting up efficiency attributes on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def set_efficiency_config_async(self):
        """Set efficiency policy and compression attributes in asynchronous mode"""
        options = {'volume-name': self.parameters['name']}
        efficiency_enable = netapp_utils.zapi.NaElement.create_node_with_children('sis-enable-async', **options)
        try:
            result = self.server.invoke_successfully(efficiency_enable, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.wrap_fail_json(msg='Error enable efficiency on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        self.check_invoke_result(result, 'enable efficiency on')
        self.set_efficiency_attributes(options)
        efficiency_start = netapp_utils.zapi.NaElement.create_node_with_children('sis-set-config-async', **options)
        try:
            result = self.server.invoke_successfully(efficiency_start, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.wrap_fail_json(msg='Error setting up efficiency attributes on volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        self.check_invoke_result(result, 'set efficiency policy on')

    def get_efficiency_info(self, return_value):
        """
        get the name of the efficiency policy assigned to volume, as well as compression values
        if attribute does not exist, set its value to None
        :return: update return_value dict.
        """
        sis_info = netapp_utils.zapi.NaElement('sis-get-iter')
        sis_status_info = netapp_utils.zapi.NaElement('sis-status-info')
        sis_status_info.add_new_child('path', '/vol/' + self.parameters['name'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(sis_status_info)
        sis_info.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(sis_info, True)
        except netapp_utils.zapi.NaApiError as error:
            if error.message.startswith('Insufficient privileges: user ') and error.message.endswith(' does not have read access to this resource'):
                self.issues.append('cannot read volume efficiency options (as expected when running as vserver): %s' % to_native(error))
                return
            self.wrap_fail_json(msg='Error fetching efficiency policy for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        for key in self.sis_keys2zapi_get:
            return_value[key] = None
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            sis_attributes = result.get_child_by_name('attributes-list').get_child_by_name('sis-status-info')
            for key, attr in self.sis_keys2zapi_get.items():
                value = sis_attributes.get_child_content(attr)
                if self.argument_spec[key]['type'] == 'bool':
                    value = self.na_helper.get_value_for_bool(True, value)
                return_value[key] = value

    def modify_volume_efficiency_config(self, efficiency_config_modify_value):
        if self.use_rest:
            return self.set_efficiency_rest()
        if efficiency_config_modify_value == 'async':
            self.set_efficiency_config_async()
        else:
            self.set_efficiency_config()

    def set_snapshot_auto_delete(self):
        options = {'volume': self.parameters['name']}
        desired_options = self.parameters['snapshot_auto_delete']
        for key, value in desired_options.items():
            options['option-name'] = key
            options['option-value'] = str(value)
            snapshot_auto_delete = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-autodelete-set-option', **options)
            try:
                self.server.invoke_successfully(snapshot_auto_delete, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.wrap_fail_json(msg='Error setting snapshot auto delete options for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def rehost_volume(self):
        volume_rehost = netapp_utils.zapi.NaElement.create_node_with_children('volume-rehost', **{'vserver': self.parameters['from_vserver'], 'destination-vserver': self.parameters['vserver'], 'volume': self.parameters['name']})
        if self.parameters.get('auto_remap_luns') is not None:
            volume_rehost.add_new_child('auto-remap-luns', str(self.parameters['auto_remap_luns']))
        if self.parameters.get('force_unmap_luns') is not None:
            volume_rehost.add_new_child('force-unmap-luns', str(self.parameters['force_unmap_luns']))
        try:
            self.cluster.invoke_successfully(volume_rehost, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error rehosting volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def snapshot_restore_volume(self):
        if self.use_rest:
            return self.snapshot_restore_volume_rest()
        snapshot_restore = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-restore-volume', **{'snapshot': self.parameters['snapshot_restore'], 'volume': self.parameters['name']})
        if self.parameters.get('force_restore') is not None:
            snapshot_restore.add_new_child('force', str(self.parameters['force_restore']))
        if self.parameters.get('preserve_lun_ids') is not None:
            snapshot_restore.add_new_child('preserve-lun-ids', str(self.parameters['preserve_lun_ids']))
        try:
            self.server.invoke_successfully(snapshot_restore, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error restoring volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def ignore_small_change(self, current, attribute, threshold):
        if attribute in current and current[attribute] != 0 and (self.parameters.get(attribute) is not None):
            change = abs(current[attribute] - self.parameters[attribute]) * 100.0 / current[attribute]
            if change < threshold:
                self.parameters[attribute] = current[attribute]
                if change > 0.1:
                    self.module.warn('resize request for %s ignored: %.1f%% is below the threshold: %.1f%%' % (attribute, change, threshold))

    def adjust_sizes(self, current, after_create):
        """
        ignore small change in size by resetting expectations
        """
        if after_create:
            self.parameters['size'] = current['size']
            return
        self.ignore_small_change(current, 'size', self.parameters['size_change_threshold'])
        self.ignore_small_change(current, 'max_files', netapp_utils.get_feature(self.module, 'max_files_change_threshold'))

    def validate_snaplock_changes(self, current, modify=None, after_create=False):
        if not self.use_rest:
            return
        msg = None
        if modify:
            if 'type' in modify['snaplock']:
                msg = 'Error: volume snaplock type was not set properly at creation time.' if after_create else 'Error: changing a volume snaplock type after creation is not allowed.'
                msg += '  Current: %s, desired: %s.' % (current['snaplock']['type'], self.parameters['snaplock']['type'])
        elif self.parameters['state'] == 'present':
            sl_dict = self.na_helper.filter_out_none_entries(self.parameters.get('snaplock', {}))
            sl_type = sl_dict.pop('type', 'non_snaplock')
            if sl_dict and (current is None and sl_type == 'non_snaplock' or (current and current['snaplock']['type'] == 'non_snaplock')):
                msg = 'Error: snaplock options are not supported for non_snaplock volume, found: %s.' % sl_dict
            if not self.rest_api.meets_rest_minimum_version(True, 9, 10, 1):
                if sl_type == 'non_snaplock':
                    self.parameters.pop('snaplock', None)
                else:
                    msg = 'Error: %s' % self.rest_api.options_require_ontap_version('snaplock type', '9.10.1', True)
        if msg:
            self.module.fail_json(msg=msg)

    def set_modify_dict(self, current, after_create=False):
        """Fill modify dict with changes"""
        octal_value = current.get('unix_permissions') if current else None
        if self.parameters.get('unix_permissions') is not None and self.na_helper.compare_chmod_value(octal_value, self.parameters['unix_permissions']):
            del self.parameters['unix_permissions']
        auto_delete_info = current.pop('snapshot_auto_delete', None)
        self.adjust_sizes(current, after_create)
        if 'type' in self.parameters:
            self.parameters['type'] = self.parameters['type'].lower()
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if modify is not None and 'type' in modify:
            msg = 'Error: volume type was not set properly at creation time.' if after_create else 'Error: changing a volume from one type to another is not allowed.'
            msg += '  Current: %s, desired: %s.' % (current['type'], self.parameters['type'])
            self.module.fail_json(msg=msg)
        if modify is not None and 'snaplock' in modify:
            self.validate_snaplock_changes(current, modify, after_create)
        desired_style = self.get_volume_style(None)
        if desired_style is not None and desired_style != self.volume_style:
            msg = 'Error: volume backend was not set properly at creation time.' if after_create else 'Error: changing a volume from one backend to another is not allowed.'
            msg += '  Current: %s, desired: %s.' % (self.volume_style, desired_style)
            self.module.fail_json(msg=msg)
        desired_tcontrol = self.na_helper.safe_get(self.parameters, ['nas_application_template', 'tiering', 'control'])
        if desired_tcontrol in ('required', 'disallowed'):
            warn_or_fail = netapp_utils.get_feature(self.module, 'warn_or_fail_on_fabricpool_backend_change')
            if warn_or_fail in ('warn', 'fail'):
                current_tcontrol = self.tiering_control(current)
                if desired_tcontrol != current_tcontrol:
                    msg = 'Error: volume tiering control was not set properly at creation time.' if after_create else 'Error: changing a volume from one backend to another is not allowed.'
                    msg += '  Current tiering control: %s, desired: %s.' % (current_tcontrol, desired_tcontrol)
                    if warn_or_fail == 'fail':
                        self.module.fail_json(msg=msg)
                    self.module.warn('Ignored ' + msg)
            elif warn_or_fail not in (None, 'ignore'):
                self.module.warn("Unexpected value '%s' for warn_or_fail_on_fabricpool_backend_change, expecting: None, 'ignore', 'fail', 'warn'" % warn_or_fail)
        if self.parameters.get('snapshot_auto_delete') is not None:
            auto_delete_modify = self.na_helper.get_modified_attributes(auto_delete_info, self.parameters['snapshot_auto_delete'])
            if len(auto_delete_modify) > 0:
                modify['snapshot_auto_delete'] = auto_delete_modify
        return modify

    def take_modify_actions(self, modify):
        self.modify_volume(modify)
        if any((modify.get(key) is not None for key in self.sis_keys2zapi_get)):
            if self.parameters.get('is_infinite') or self.volume_style == 'flexgroup':
                efficiency_config_modify = 'async'
            else:
                efficiency_config_modify = 'sync'
            self.modify_volume_efficiency_config(efficiency_config_modify)
        if modify.get('is_online') is False:
            self.change_volume_state()
    " MAPPING OF VOLUME FIELDS FROM ZAPI TO REST\n    ZAPI = REST\n    encrypt = encryption.enabled\n    volume-comp-aggr-attributes.tiering-policy = tiering.policy\n    'volume-export-attributes.policy' = nas.export_policy.name\n    'volume-id-attributes.containing-aggregate-name' = aggregates.name\n    'volume-id-attributes.flexgroup-uuid' = uuid (Only for FlexGroup volumes)\n    'volume-id-attributes.instance-uuid' = uuid (Only for FlexVols)\n    'volume-id-attributes.junction-path' = nas.path\n    'volume-id-attributes.style-extended' = style\n    'volume-id-attributes.type' = type\n    'volume-id-attributes.comment' = comment\n    'volume-performance-attributes.is-atime-update-enabled' == NO REST VERSION\n    volume-qos-attributes.policy-group-name' = qos.policy.name\n    'volume-qos-attributes.adaptive-policy-group-name' = qos.policy.name\n    'volume-security-attributes.style = nas.security_style\n    volume-security-attributes.volume-security-unix-attributes.group-id' = nas.gid\n    'volume-security-attributes.volume-security-unix-attributes.permissions' =  nas.unix_permissions\n    'volume-security-attributes.volume-security-unix-attributes.user-id' = nas.uid\n    'volume-snapshot-attributes.snapdir-access-enabled' == NO REST VERSION\n    'volume-snapshot-attributes,snapshot-policy' = snapshot_policy\n    volume-space-attributes.percentage-snapshot-reserve = space.snapshot.reserve_percent\n    volume-space-attributes.size' = space.size\n    'volume-space-attributes.space-guarantee' = guarantee.type\n    volume-space-attributes.space-slo' == NO REST VERSION\n    'volume-state-attributes.is-nvfail-enabled' == NO REST Version\n    'volume-state-attributes.state' = state\n    'volume-vserver-dr-protection-attributes.vserver-dr-protection' = == NO REST Version\n    volume-snapshot-autodelete-attributes.* None exist other than space.snapshot.autodelete_enabled\n    From get_efficiency_info function\n    efficiency_policy = efficiency.policy.name\n    compression = efficiency.compression\n    inline_compression = efficiency.compression\n    "

    def get_volume_rest(self, vol_name):
        """
        This covers the zapi functions
        get_volume
         - volume_get_iter
         - get_efficiency_info
        """
        api = 'storage/volumes'
        params = {'name': vol_name, 'svm.name': self.parameters['vserver'], 'fields': 'encryption.enabled,tiering.policy,nas.export_policy.name,aggregates.name,aggregates.uuid,uuid,nas.path,style,type,comment,qos.policy.name,nas.security_style,nas.gid,nas.unix_permissions,nas.uid,snapshot_policy,space.snapshot.reserve_percent,space.size,guarantee.type,state,efficiency.compression,snaplock,files.maximum,space.logical_space.enforcement,space.logical_space.reporting,'}
        if self.parameters.get('efficiency_policy'):
            params['fields'] += 'efficiency.policy.name,'
        if self.parameters.get('tiering_minimum_cooling_days'):
            params['fields'] += 'tiering.min_cooling_days,'
        if self.parameters.get('analytics'):
            params['fields'] += 'analytics,'
        if self.parameters.get('tags'):
            params['fields'] += '_tags,'
        if self.parameters.get('atime_update') is not None:
            params['fields'] += 'access_time_enabled,'
        if self.parameters.get('snapdir_access') is not None:
            params['fields'] += 'snapshot_directory_access_enabled,'
        if self.parameters.get('snapshot_auto_delete') is not None:
            params['fields'] += 'space.snapshot.autodelete,'
        if self.parameters.get('vol_nearly_full_threshold_percent') is not None:
            params['fields'] += 'space.nearly_full_threshold_percent,'
        if self.parameters.get('vol_full_threshold_percent') is not None:
            params['fields'] += 'space.full_threshold_percent,'
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg=error)
        return self.format_get_volume_rest(record) if record else None

    def rename_volume_rest(self):
        current = self.get_volume_rest(self.parameters['from_name'])
        body = {'name': self.parameters['name']}
        dummy, error = self.volume_rest_patch(body, uuid=current['uuid'])
        if error:
            self.module.fail_json(msg='Error changing name of volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def snapshot_restore_volume_rest(self):
        current = self.get_volume()
        self.parameters['uuid'] = current['uuid']
        body = {'restore_to.snapshot.name': self.parameters['snapshot_restore']}
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error restoring snapshot %s in volume %s: %s' % (self.parameters['snapshot_restore'], self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def create_volume_rest(self):
        body = self.create_volume_body_rest()
        dummy, error = rest_generic.post_async(self.rest_api, 'storage/volumes', body, job_timeout=self.parameters['time_out'])
        if error:
            self.module.fail_json(msg='Error creating volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_online(sleep_time=5)

    def create_volume_body_rest(self):
        body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
        if self.parameters.get('space_guarantee') is not None:
            body['guarantee.type'] = self.parameters['space_guarantee']
        body = self.aggregates_rest(body)
        if self.parameters.get('tags') is not None:
            body['_tags'] = self.parameters['tags']
        if self.parameters.get('size') is not None:
            body['size'] = self.parameters['size']
        if self.parameters.get('snapshot_policy') is not None:
            body['snapshot_policy.name'] = self.parameters['snapshot_policy']
        if self.parameters.get('unix_permissions') is not None:
            body['nas.unix_permissions'] = self.parameters['unix_permissions']
        if self.parameters.get('group_id') is not None:
            body['nas.gid'] = self.parameters['group_id']
        if self.parameters.get('user_id') is not None:
            body['nas.uid'] = self.parameters['user_id']
        if self.parameters.get('volume_security_style') is not None:
            body['nas.security_style'] = self.parameters['volume_security_style']
        if self.parameters.get('export_policy') is not None:
            body['nas.export_policy.name'] = self.parameters['export_policy']
        if self.parameters.get('junction_path') is not None:
            body['nas.path'] = self.parameters['junction_path']
        if self.parameters.get('comment') is not None:
            body['comment'] = self.parameters['comment']
        if self.parameters.get('type') is not None:
            body['type'] = self.parameters['type'].lower()
        if self.parameters.get('percent_snapshot_space') is not None:
            body['space.snapshot.reserve_percent'] = self.parameters['percent_snapshot_space']
        if self.parameters.get('language') is not None:
            body['language'] = self.parameters['language']
        if self.get_qos_policy_group() is not None:
            body['qos.policy.name'] = self.get_qos_policy_group()
        if self.parameters.get('tiering_policy') is not None:
            body['tiering.policy'] = self.parameters['tiering_policy']
        if self.parameters.get('encrypt') is not None:
            body['encryption.enabled'] = self.parameters['encrypt']
        if self.parameters.get('logical_space_enforcement') is not None:
            body['space.logical_space.enforcement'] = self.parameters['logical_space_enforcement']
        if self.parameters.get('logical_space_reporting') is not None:
            body['space.logical_space.reporting'] = self.parameters['logical_space_reporting']
        if self.parameters.get('tiering_minimum_cooling_days') is not None:
            body['tiering.min_cooling_days'] = self.parameters['tiering_minimum_cooling_days']
        if self.parameters.get('snaplock') is not None:
            body['snaplock'] = self.na_helper.filter_out_none_entries(self.parameters['snaplock'])
        if self.volume_style:
            body['style'] = self.volume_style
        if self.parameters.get('efficiency_policy') is not None:
            body['efficiency.policy.name'] = self.parameters['efficiency_policy']
        if self.get_compression():
            body['efficiency.compression'] = self.get_compression()
        if self.parameters.get('analytics'):
            body['analytics.state'] = self.parameters['analytics']
        body['state'] = self.bool_to_online(self.parameters['is_online'])
        return body

    def aggregates_rest(self, body):
        if self.parameters.get('aggregate_name') is not None:
            body['aggregates'] = [{'name': self.parameters['aggregate_name']}]
        if self.parameters.get('aggr_list') is not None:
            body['aggregates'] = [{'name': name} for name in self.parameters['aggr_list']]
        if self.parameters.get('aggr_list_multiplier') is not None:
            body['constituents_per_aggregate'] = self.parameters['aggr_list_multiplier']
        return body

    def volume_modify_attributes_rest(self, params):
        body = self.modify_volume_body_rest(params)
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error modifying volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    @staticmethod
    def bool_to_online(item):
        return 'online' if item else 'offline'

    @staticmethod
    def enabled_to_bool(item, reverse=False):
        """ convertes on/off to true/false or vice versa """
        if reverse:
            return 'on' if item else 'off'
        return True if item == 'on' else False

    def modify_volume_body_rest(self, params):
        body = {}
        for key, option, transform in [('analytics.state', 'analytics', None), ('guarantee.type', 'space_guarantee', None), ('space.snapshot.reserve_percent', 'percent_snapshot_space', None), ('snapshot_policy.name', 'snapshot_policy', None), ('nas.export_policy.name', 'export_policy', None), ('nas.unix_permissions', 'unix_permissions', None), ('nas.gid', 'group_id', None), ('nas.uid', 'user_id', None), ('qos.policy.name', 'qos_policy_group', None), ('qos.policy.name', 'qos_adaptive_policy_group', None), ('comment', 'comment', None), ('space.logical_space.enforcement', 'logical_space_enforcement', None), ('space.logical_space.reporting', 'logical_space_reporting', None), ('tiering.min_cooling_days', 'tiering_minimum_cooling_days', None), ('state', 'is_online', self.bool_to_online), ('_tags', 'tags', None), ('snapshot_directory_access_enabled', 'snapdir_access', None), ('access_time_enabled', 'atime_update', None), ('space.nearly_full_threshold_percent', 'vol_nearly_full_threshold_percent', None), ('space.full_threshold_percent', 'vol_full_threshold_percent', None)]:
            value = self.parameters.get(option)
            if value is not None and transform:
                value = transform(value)
            if value is not None:
                body[key] = value
        for key, option, transform in [('nas.security_style', 'volume_security_style', None), ('tiering.policy', 'tiering_policy', None), ('files.maximum', 'max_files', None)]:
            if params and params.get(option) is not None:
                body[key] = self.parameters[option]
        if params and params.get('snaplock') is not None:
            sl_dict = self.na_helper.filter_out_none_entries(self.parameters['snaplock']) or {}
            sl_dict.pop('type', None)
            if sl_dict:
                body['snaplock'] = sl_dict
        if params and params.get('snapshot_auto_delete') is not None:
            for key, option, transform in [('space.snapshot.autodelete.trigger', 'trigger', None), ('space.snapshot.autodelete.target_free_space', 'target_free_space', None), ('space.snapshot.autodelete.delete_order', 'delete_order', None), ('space.snapshot.autodelete.commitment', 'commitment', None), ('space.snapshot.autodelete.defer_delete', 'defer_delete', None), ('space.snapshot.autodelete.prefix', 'prefix', None), ('space.snapshot.autodelete.enabled', 'state', self.enabled_to_bool)]:
                if params and params['snapshot_auto_delete'].get(option) is not None:
                    if transform:
                        body[key] = transform(self.parameters['snapshot_auto_delete'][option])
                    else:
                        body[key] = self.parameters['snapshot_auto_delete'][option]
        return body

    def change_volume_state_rest(self):
        body = {'state': self.bool_to_online(self.parameters['is_online'])}
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error changing state of volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return (body['state'], None)

    def volume_unmount_rest(self, fail_on_error=True):
        body = {'nas.path': ''}
        dummy, error = self.volume_rest_patch(body)
        if error and fail_on_error:
            self.module.fail_json(msg='Error unmounting volume %s with path "%s": %s' % (self.parameters['name'], self.parameters.get('junction_path'), to_native(error)), exception=traceback.format_exc())
        return error

    def volume_mount_rest(self):
        body = {'nas.path': self.parameters['junction_path']}
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error mounting volume %s with path "%s": %s' % (self.parameters['name'], self.parameters['junction_path'], to_native(error)), exception=traceback.format_exc())

    def set_efficiency_rest(self):
        body = {}
        if self.parameters.get('efficiency_policy') is not None:
            body['efficiency.policy.name'] = self.parameters['efficiency_policy']
        if self.get_compression():
            body['efficiency.compression'] = self.get_compression()
        if not body:
            return
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error setting efficiency for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def encryption_conversion_rest(self):
        body = {'encryption.enabled': True}
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error enabling encryption for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_encryption_conversion_rest()

    def resize_volume_rest(self):
        query = None
        if self.parameters.get('sizing_method') is not None:
            query = dict(sizing_method=self.parameters['sizing_method'])
        body = {'size': self.parameters['size']}
        dummy, error = self.volume_rest_patch(body, query)
        if error:
            self.module.fail_json(msg='Error resizing volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def move_volume_rest(self, encrypt_destination):
        body = {'movement.destination_aggregate.name': self.parameters['aggregate_name']}
        if encrypt_destination is not None:
            body['encryption.enabled'] = encrypt_destination
        dummy, error = self.volume_rest_patch(body)
        if error:
            self.module.fail_json(msg='Error moving volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_move_rest()

    def volume_rest_patch(self, body, query=None, uuid=None):
        if not uuid:
            uuid = self.parameters['uuid']
        if not uuid:
            self.module.fail_json(msg='Could not read UUID for volume %s in patch.' % self.parameters['name'])
        return rest_generic.patch_async(self.rest_api, 'storage/volumes', uuid, body, query=query, job_timeout=self.parameters['time_out'])

    def get_qos_policy_group(self):
        if self.parameters.get('qos_policy_group') is not None:
            return self.parameters['qos_policy_group']
        if self.parameters.get('qos_adaptive_policy_group') is not None:
            return self.parameters['qos_adaptive_policy_group']
        return None

    def get_compression(self):
        if self.parameters.get('compression') and self.parameters.get('inline_compression'):
            return 'both'
        if self.parameters.get('compression'):
            return 'background'
        if self.parameters.get('inline_compression'):
            return 'inline'
        if self.parameters.get('compression') is False and self.parameters.get('inline_compression') is False:
            return 'none'
        return None

    def rest_errors(self):
        if self.parameters.get('qos_policy_group') and self.parameters.get('qos_adaptive_policy_group'):
            self.module.fail_json(msg='Error: With Rest API qos_policy_group and qos_adaptive_policy_group are now the same thing, and cannot be set at the same time')
        ontap_97_options = ['nas_application_template']
        if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 7) and any((x in self.parameters for x in ontap_97_options)):
            self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_97_options, version='9.7'))
        if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9) and self.na_helper.safe_get(self.parameters, ['nas_application_template', 'flexcache', 'dr_cache']) is not None:
            self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version('flexcache: dr_cache', version='9.9'))
        if 'snapshot_auto_delete' in self.parameters:
            if 'destroy_list' in self.parameters['snapshot_auto_delete']:
                self.module.fail_json(msg="snapshot_auto_delete option 'destroy_list' is currently not supported with REST.")

    def format_get_volume_rest(self, record):
        is_online = record.get('state') == 'online'
        aggregates = record.get('aggregates', None)
        aggr_name = aggregates[0].get('name', None) if aggregates else None
        rest_compression = self.na_helper.safe_get(record, ['efficiency', 'compression'])
        junction_path = self.na_helper.safe_get(record, ['nas', 'path'])
        if junction_path is None:
            junction_path = ''
        state = self.na_helper.safe_get(record, ['analytics', 'state'])
        analytics = 'on' if state == 'initializing' else state
        auto_delete_info = self.na_helper.safe_get(record, ['space', 'snapshot', 'autodelete'])
        if auto_delete_info is not None:
            auto_delete_info['state'] = self.enabled_to_bool(self.na_helper.safe_get(record, ['space', 'snapshot', 'autodelete', 'enabled']), reverse=True)
            del auto_delete_info['enabled']
        return {'tags': record.get('_tags', []), 'name': record.get('name', None), 'analytics': analytics, 'encrypt': self.na_helper.safe_get(record, ['encryption', 'enabled']), 'tiering_policy': self.na_helper.safe_get(record, ['tiering', 'policy']), 'export_policy': self.na_helper.safe_get(record, ['nas', 'export_policy', 'name']), 'aggregate_name': aggr_name, 'aggregates': aggregates, 'flexgroup_uuid': record.get('uuid', None), 'instance_uuid': record.get('uuid', None), 'junction_path': junction_path, 'style_extended': record.get('style', None), 'type': record.get('type', None), 'comment': record.get('comment', None), 'qos_policy_group': self.na_helper.safe_get(record, ['qos', 'policy', 'name']), 'qos_adaptive_policy_group': self.na_helper.safe_get(record, ['qos', 'policy', 'name']), 'volume_security_style': self.na_helper.safe_get(record, ['nas', 'security_style']), 'group_id': self.na_helper.safe_get(record, ['nas', 'gid']), 'unix_permissions': str(self.na_helper.safe_get(record, ['nas', 'unix_permissions'])), 'user_id': self.na_helper.safe_get(record, ['nas', 'uid']), 'snapshot_policy': self.na_helper.safe_get(record, ['snapshot_policy', 'name']), 'percent_snapshot_space': self.na_helper.safe_get(record, ['space', 'snapshot', 'reserve_percent']), 'size': self.na_helper.safe_get(record, ['space', 'size']), 'space_guarantee': self.na_helper.safe_get(record, ['guarantee', 'type']), 'is_online': is_online, 'uuid': record.get('uuid', None), 'efficiency_policy': self.na_helper.safe_get(record, ['efficiency', 'policy', 'name']), 'compression': rest_compression in ('both', 'background'), 'inline_compression': rest_compression in ('both', 'inline'), 'logical_space_enforcement': self.na_helper.safe_get(record, ['space', 'logical_space', 'enforcement']), 'logical_space_reporting': self.na_helper.safe_get(record, ['space', 'logical_space', 'reporting']), 'tiering_minimum_cooling_days': self.na_helper.safe_get(record, ['tiering', 'min_cooling_days']), 'snaplock': self.na_helper.safe_get(record, ['snaplock']), 'max_files': self.na_helper.safe_get(record, ['files', 'maximum']), 'atime_update': record.get('access_time_enabled', True), 'snapdir_access': record.get('snapshot_directory_access_enabled', True), 'snapshot_auto_delete': auto_delete_info, 'vol_nearly_full_threshold_percent': self.na_helper.safe_get(record, ['space', 'nearly_full_threshold_percent']), 'vol_full_threshold_percent': self.na_helper.safe_get(record, ['space', 'full_threshold_percent'])}

    def is_fabricpool(self, name, aggregate_uuid):
        """whether the aggregate is associated with one or more object stores"""
        api = 'storage/aggregates/%s/cloud-stores' % aggregate_uuid
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api)
        if error:
            self.module.fail_json(msg='Error getting object store for aggregate: %s: %s' % (name, error))
        return records is not None and len(records) > 0

    def tiering_control(self, current):
        """return whether the backend meets FabricPool requirements:
            required: all aggregates are in a FabricPool
            disallowed: all aggregates are not in a FabricPool
            best_effort: mixed
        """
        fabricpools = [self.is_fabricpool(aggregate['name'], aggregate['uuid']) for aggregate in current.get('aggregates', [])]
        if not fabricpools:
            return None
        if all(fabricpools):
            return 'required'
        if any(fabricpools):
            return 'best_effort'
        return 'disallowed'

    def set_actions(self):
        """define what needs to be done"""
        actions = []
        modify = {}
        current = self.get_volume()
        self.volume_style = self.get_volume_style(current)
        if self.volume_style == 'flexgroup' and self.parameters.get('aggregate_name') is not None:
            self.module.fail_json(msg='Error: aggregate_name option cannot be used with FlexGroups.')
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'delete' or self.parameters['state'] == 'absent':
            return (['delete'] if cd_action == 'delete' else [], current, modify)
        if cd_action == 'create':
            if self.use_rest:
                rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)
            actions = ['create']
            if self.parameters.get('from_name'):
                current = self.get_volume(self.parameters['from_name'])
                rename = self.na_helper.is_rename_action(current, None)
                if rename is None:
                    self.module.fail_json(msg='Error renaming volume: cannot find %s' % self.parameters['from_name'])
                if rename:
                    cd_action = None
                    actions = ['rename']
            elif self.parameters.get('from_vserver'):
                if self.use_rest:
                    self.module.fail_json(msg='Error: ONTAP REST API does not support Rehosting Volumes')
                actions = ['rehost']
                self.na_helper.changed = True
        if self.parameters.get('snapshot_restore'):
            if 'create' in actions:
                self.module.fail_json(msg='Error restoring volume: cannot find parent: %s' % self.parameters['name'])
            actions.append('snapshot_restore')
            self.na_helper.changed = True
        self.validate_snaplock_changes(current)
        if cd_action is None and 'rehost' not in actions:
            modify = self.set_modify_dict(current)
            if modify:
                if not self.use_rest and modify.get('encrypt') is False and (not self.parameters.get('aggregate_name')):
                    self.parameters['aggregate_name'] = current['aggregate_name']
                if self.use_rest and modify.get('encrypt') is False and (not modify.get('aggregate_name')):
                    self.module.fail_json(msg='Error: unencrypting volume is only supported when moving the volume to another aggregate in REST.')
                actions.append('modify')
        if self.parameters.get('nas_application_template') is not None:
            application = self.get_application()
            changed = self.na_helper.changed
            app_component = self.create_nas_application_component() if self.parameters['state'] == 'present' else None
            modify_app = self.na_helper.get_modified_attributes(application, app_component)
            if modify_app:
                self.na_helper.changed = changed
                self.module.warn('Modifying an app is not supported at present: ignoring: %s' % str(modify_app))
        return (actions, current, modify)

    def apply(self):
        """Call create/modify/delete operations"""
        actions, current, modify = self.set_actions()
        is_online = current.get('is_online') if current else None
        response = None
        online_modify_options = [x for x in actions if x in ['rehost', 'snapshot_restore', 'modify']]
        if not modify.get('is_online') and is_online is False and online_modify_options:
            modify_keys = []
            if 'modify' in online_modify_options:
                online_modify_options.remove('modify')
                modify_keys = [key for key in modify if key != 'is_online']
            action_msg = 'perform action(s): %s' % online_modify_options if online_modify_options else ''
            modify_msg = ' and modify: %s' % modify_keys if action_msg else 'modify: %s' % modify_keys
            self.module.warn('Cannot %s%s when volume is offline.' % (action_msg, modify_msg))
            modify, actions = ({}, [])
            if 'rename' in actions:
                actions = ['rename']
            else:
                self.na_helper.changed = False
        if self.na_helper.changed and (not self.module.check_mode):
            if modify.get('is_online'):
                self.parameters['uuid'] = current['uuid']
                for field in ['volume_security_style', 'group_id', 'user_id', 'percent_snapshot_space']:
                    if self.parameters.get(field) is not None:
                        modify[field] = self.parameters[field]
                self.change_volume_state()
            if 'rename' in actions:
                self.rename_volume()
            if 'rehost' in actions:
                self.rehost_volume()
            if 'snapshot_restore' in actions:
                self.snapshot_restore_volume()
            if 'create' in actions:
                response = self.create_volume()
                current = self.get_volume()
                if current:
                    self.volume_created = True
                    modify = self.set_modify_dict(current, after_create=True)
                    is_online = current.get('is_online')
                    if modify:
                        if is_online:
                            actions.append('modify')
                        else:
                            self.module.warn('Cannot perform actions: modify when volume is offline.')
                self.na_helper.changed = True
            if 'delete' in actions:
                self.parameters['uuid'] = current['uuid']
                self.delete_volume(current)
            if 'modify' in actions:
                self.parameters['uuid'] = current['uuid']
                self.take_modify_actions(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, actions, modify, response)
        self.module.exit_json(**result)