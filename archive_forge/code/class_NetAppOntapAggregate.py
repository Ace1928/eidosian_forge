from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
class NetAppOntapAggregate:
    """ object initialize and class methods """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=True, type='str'), disks=dict(required=False, type='list', elements='str'), disk_count=dict(required=False, type='int', default=None), disk_size=dict(required=False, type='int'), disk_size_with_unit=dict(required=False, type='str'), disk_class=dict(required=False, choices=['capacity', 'performance', 'archive', 'solid_state', 'array', 'virtual', 'data_center', 'capacity_flash']), disk_type=dict(required=False, choices=['ATA', 'BSAS', 'FCAL', 'FSAS', 'LUN', 'MSATA', 'SAS', 'SSD', 'SSD-CAP', 'SSD-NVM', 'VMDISK', 'VMLUN', 'VMLUN-SSD']), from_name=dict(required=False, type='str'), mirror_disks=dict(required=False, type='list', elements='str'), nodes=dict(required=False, type='list', elements='str'), is_mirrored=dict(required=False, type='bool'), raid_size=dict(required=False, type='int'), raid_type=dict(required=False, choices=['raid4', 'raid_dp', 'raid_tec', 'raid_0']), service_state=dict(required=False, choices=['online', 'offline']), spare_pool=dict(required=False, choices=['Pool0', 'Pool1']), state=dict(required=False, choices=['present', 'absent'], default='present'), unmount_volumes=dict(required=False, type='bool'), wait_for_online=dict(required=False, type='bool', default=False), time_out=dict(required=False, type='int', default=100), object_store_name=dict(required=False, type='str'), allow_flexgroups=dict(required=False, type='bool'), snaplock_type=dict(required=False, type='str', choices=['compliance', 'enterprise', 'non_snaplock']), ignore_pool_checks=dict(required=False, type='bool'), encryption=dict(required=False, type='bool'), tags=dict(required=False, type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, mutually_exclusive=[('is_mirrored', 'disks'), ('is_mirrored', 'mirror_disks'), ('is_mirrored', 'spare_pool'), ('spare_pool', 'disks'), ('disk_count', 'disks'), ('disk_size', 'disk_size_with_unit'), ('disk_class', 'disk_type')], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.uuid = None
        unsupported_rest_properties = ['disks', 'disk_type', 'mirror_disks', 'spare_pool', 'unmount_volumes']
        partially_supported_rest_properties = [['service_state', (9, 11, 1)], ['tags', (9, 13, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            if 'tags' in self.parameters:
                self.module.fail_json(msg='Error: tags only supported with REST.')
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)
        if self.parameters['state'] == 'present':
            self.validate_options()

    def validate_options(self):
        errors = []
        if self.use_rest:
            if len(self.parameters.get('nodes', [])) > 1:
                errors.append('only one node can be specified when using rest, found %s' % self.parameters['nodes'])
            if 'disk_count' in self.parameters and 'nodes' not in self.parameters:
                errors.append('nodes is required when disk_count is present')
        elif self.parameters.get('mirror_disks') is not None and self.parameters.get('disks') is None:
            errors.append('mirror_disks require disks options to be set')
        if errors:
            plural = 's' if len(errors) > 1 else ''
            self.module.fail_json(msg='Error%s when validating options: %s.' % (plural, '; '.join(errors)))

    def aggr_get_iter(self, name):
        """
        Return aggr-get-iter query results
        :param name: Name of the aggregate
        :return: NaElement if aggregate found, None otherwise
        """
        aggr_get_iter = netapp_utils.zapi.NaElement('aggr-get-iter')
        query_details = netapp_utils.zapi.NaElement.create_node_with_children('aggr-attributes', **{'aggregate-name': name})
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(query_details)
        aggr_get_iter.add_child_elem(query)
        result = None
        try:
            result = self.server.invoke_successfully(aggr_get_iter, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) != '13040':
                self.module.fail_json(msg='Error getting aggregate: %s' % to_native(error), exception=traceback.format_exc())
        return result

    def get_aggr(self, name=None):
        """
        Fetch details if aggregate exists.
        :param name: Name of the aggregate to be fetched
        :return:
            Dictionary of current details if aggregate found
            None if aggregate is not found
        """
        if name is None:
            name = self.parameters.get('name')
        if self.use_rest:
            return self.get_aggr_rest(name)
        aggr_get = self.aggr_get_iter(name)
        if aggr_get and aggr_get.get_child_by_name('num-records') and (int(aggr_get.get_child_content('num-records')) >= 1):
            attr = aggr_get.get_child_by_name('attributes-list').get_child_by_name('aggr-attributes')
            current_aggr = {'service_state': attr.get_child_by_name('aggr-raid-attributes').get_child_content('state')}
            if attr.get_child_by_name('aggr-raid-attributes').get_child_content('disk-count'):
                current_aggr['disk_count'] = int(attr.get_child_by_name('aggr-raid-attributes').get_child_content('disk-count'))
            if attr.get_child_by_name('aggr-raid-attributes').get_child_content('encrypt-with-aggr-key'):
                current_aggr['encryption'] = attr.get_child_by_name('aggr-raid-attributes').get_child_content('encrypt-with-aggr-key') == 'true'
            snaplock_type = self.na_helper.safe_get(attr, ['aggr-snaplock-attributes', 'snaplock-type'])
            if snaplock_type:
                current_aggr['snaplock_type'] = snaplock_type
            return current_aggr
        return None

    def disk_get_iter(self, name):
        """
        Return storage-disk-get-iter query results
        Filter disk list by aggregate name, and only reports disk-name and plex-name
        :param name: Name of the aggregate
        :return: NaElement
        """
        disk_get_iter = netapp_utils.zapi.NaElement('storage-disk-get-iter')
        query_details = {'query': {'storage-disk-info': {'disk-raid-info': {'disk-aggregate-info': {'aggregate-name': name}}}}}
        disk_get_iter.translate_struct(query_details)
        attributes = {'desired-attributes': {'storage-disk-info': {'disk-name': None, 'disk-raid-info': {'disk_aggregate_info': {'plex-name': None}}}}}
        disk_get_iter.translate_struct(attributes)
        result = None
        try:
            result = self.server.invoke_successfully(disk_get_iter, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting disks: %s' % to_native(error), exception=traceback.format_exc())
        return result

    def get_aggr_disks(self, name):
        """
        Fetch disks that are used for this aggregate.
        :param name: Name of the aggregate to be fetched
        :return:
            list of tuples (disk-name, plex-name)
            empty list if aggregate is not found
        """
        disks = []
        aggr_get = self.disk_get_iter(name)
        if aggr_get and aggr_get.get_child_by_name('num-records') and (int(aggr_get.get_child_content('num-records')) >= 1):
            attr = aggr_get.get_child_by_name('attributes-list')
            disks = [(disk_info.get_child_content('disk-name'), disk_info.get_child_by_name('disk-raid-info').get_child_by_name('disk-aggregate-info').get_child_content('plex-name')) for disk_info in attr.get_children()]
        return disks

    def object_store_get_iter(self, name):
        """
        Return aggr-object-store-get query results
        :return: NaElement if object-store for given aggregate found, None otherwise
        """
        object_store_get_iter = netapp_utils.zapi.NaElement('aggr-object-store-get-iter')
        query_details = netapp_utils.zapi.NaElement.create_node_with_children('object-store-information', **{'object-store-name': self.parameters.get('object_store_name'), 'aggregate': name})
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(query_details)
        object_store_get_iter.add_child_elem(query)
        result = None
        try:
            result = self.server.invoke_successfully(object_store_get_iter, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting object store: %s' % to_native(error), exception=traceback.format_exc())
        return result

    def get_object_store(self, name):
        """
        Fetch details if object store attached to the given aggregate exists.
        :return:
            Dictionary of current details if object store attached to the given aggregate is found
            None if object store is not found
        """
        if self.use_rest:
            return self.get_object_store_rest()
        object_store_get = self.object_store_get_iter(name)
        if object_store_get and object_store_get.get_child_by_name('num-records') and (int(object_store_get.get_child_content('num-records')) >= 1):
            attr = object_store_get.get_child_by_name('attributes-list').get_child_by_name('object-store-information')
            return {'object_store_name': attr.get_child_content('object-store-name')}
        return None

    def aggregate_online(self):
        """
        Set state of an offline aggregate to online
        :return: None
        """
        if self.use_rest:
            return self.patch_aggr_rest('make service state online for', {'state': 'online'})
        online_aggr = netapp_utils.zapi.NaElement.create_node_with_children('aggr-online', **{'aggregate': self.parameters['name'], 'force-online': 'true'})
        try:
            self.server.invoke_successfully(online_aggr, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error changing the state of aggregate %s to %s: %s' % (self.parameters['name'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())

    def aggregate_offline(self):
        """
        Set state of an online aggregate to offline
        :return: None
        """
        if self.use_rest:
            return self.patch_aggr_rest('make service state offline for', {'state': 'offline'})
        offline_aggr = netapp_utils.zapi.NaElement.create_node_with_children('aggr-offline', **{'aggregate': self.parameters['name'], 'force-offline': 'false', 'unmount-volumes': str(self.parameters.get('unmount_volumes', False))})
        retry = 10
        while retry > 0:
            try:
                self.server.invoke_successfully(offline_aggr, enable_tunneling=True)
                break
            except netapp_utils.zapi.NaApiError as error:
                if 'disk add operation is in progress' in to_native(error):
                    retry -= 1
                    if retry > 0:
                        continue
                self.module.fail_json(msg='Error changing the state of aggregate %s to %s: %s' % (self.parameters['name'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())

    @staticmethod
    def get_disks_or_mirror_disks_object(name, disks):
        """
        create ZAPI object for disks or mirror_disks
        """
        disks_obj = netapp_utils.zapi.NaElement(name)
        for disk in disks:
            disk_info_obj = netapp_utils.zapi.NaElement('disk-info')
            disk_info_obj.add_new_child('name', disk)
            disks_obj.add_child_elem(disk_info_obj)
        return disks_obj

    def create_aggr(self):
        """
        Create aggregate
        :return: None
        """
        if self.use_rest:
            return self.create_aggr_rest()
        options = {'aggregate': self.parameters['name']}
        if self.parameters.get('disk_class'):
            options['disk-class'] = self.parameters['disk_class']
        if self.parameters.get('disk_type'):
            options['disk-type'] = self.parameters['disk_type']
        if self.parameters.get('raid_type'):
            options['raid-type'] = self.parameters['raid_type']
        if self.parameters.get('snaplock_type'):
            options['snaplock-type'] = self.parameters['snaplock_type']
        if self.parameters.get('spare_pool'):
            options['spare-pool'] = self.parameters['spare_pool']
        if self.parameters.get('disk_count'):
            options['disk-count'] = str(self.parameters['disk_count'])
        if self.parameters.get('disk_size'):
            options['disk-size'] = str(self.parameters['disk_size'])
        if self.parameters.get('disk_size_with_unit'):
            options['disk-size-with-unit'] = str(self.parameters['disk_size_with_unit'])
        if self.parameters.get('raid_size'):
            options['raid-size'] = str(self.parameters['raid_size'])
        if self.parameters.get('is_mirrored'):
            options['is-mirrored'] = str(self.parameters['is_mirrored']).lower()
        if self.parameters.get('ignore_pool_checks'):
            options['ignore-pool-checks'] = str(self.parameters['ignore_pool_checks']).lower()
        if self.parameters.get('encryption'):
            options['encrypt-with-aggr-key'] = str(self.parameters['encryption']).lower()
        aggr_create = netapp_utils.zapi.NaElement.create_node_with_children('aggr-create', **options)
        if self.parameters.get('nodes'):
            nodes_obj = netapp_utils.zapi.NaElement('nodes')
            aggr_create.add_child_elem(nodes_obj)
            for node in self.parameters['nodes']:
                nodes_obj.add_new_child('node-name', node)
        if self.parameters.get('disks'):
            aggr_create.add_child_elem(self.get_disks_or_mirror_disks_object('disks', self.parameters.get('disks')))
        if self.parameters.get('mirror_disks'):
            aggr_create.add_child_elem(self.get_disks_or_mirror_disks_object('mirror-disks', self.parameters.get('mirror_disks')))
        try:
            self.server.invoke_successfully(aggr_create, enable_tunneling=False)
            if self.parameters.get('wait_for_online'):
                retries = (self.parameters['time_out'] + 5) / 10
                current = self.get_aggr()
                status = None if current is None else current['service_state']
                while status != 'online' and retries > 0:
                    time.sleep(10)
                    retries = retries - 1
                    current = self.get_aggr()
                    status = None if current is None else current['service_state']
            else:
                current = self.get_aggr()
            if current is not None and current.get('disk_count') != self.parameters.get('disk_count'):
                self.module.warn('Aggregate created with mismatched disk_count: created %s not %s' % (current.get('disk_count'), self.parameters.get('disk_count')))
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error provisioning aggregate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_aggr(self):
        """
        Delete aggregate.
        :return: None
        """
        if self.use_rest:
            return self.delete_aggr_rest()
        aggr_destroy = netapp_utils.zapi.NaElement.create_node_with_children('aggr-destroy', **{'aggregate': self.parameters['name']})
        try:
            self.server.invoke_successfully(aggr_destroy, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error removing aggregate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def rename_aggregate(self):
        """
        Rename aggregate.
        """
        if self.use_rest:
            return self.rename_aggr_rest()
        aggr_rename = netapp_utils.zapi.NaElement.create_node_with_children('aggr-rename', **{'aggregate': self.parameters['from_name'], 'new-aggregate-name': self.parameters['name']})
        try:
            self.server.invoke_successfully(aggr_rename, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming aggregate %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())

    def modify_aggr(self, modify):
        """
        Modify state of the aggregate
        :param modify: dictionary of parameters to be modified
        :return: None
        """
        if modify.get('service_state') == 'online':
            self.aggregate_online()
        if modify.get('tags') is not None:
            self.patch_aggr_rest('modify tags for', {'_tags': modify['tags']})
        disk_size = self.parameters.get('disk_size', 0)
        disk_size_with_unit = self.parameters.get('disk_size_with_unit')
        if modify.get('disk_count'):
            self.add_disks(modify['disk_count'], disk_size=disk_size, disk_size_with_unit=disk_size_with_unit)
        if modify.get('disks_to_add') or modify.get('mirror_disks_to_add'):
            self.add_disks(0, modify.get('disks_to_add'), modify.get('mirror_disks_to_add'))
        if modify.get('service_state') == 'offline':
            self.aggregate_offline()

    def attach_object_store_to_aggr(self):
        """
        Attach object store to aggregate.
        :return: None
        """
        if self.use_rest:
            return self.attach_object_store_to_aggr_rest()
        store_obj = {'aggregate': self.parameters['name'], 'object-store-name': self.parameters['object_store_name']}
        if 'allow_flexgroups' in self.parameters:
            store_obj['allow-flexgroup'] = self.na_helper.get_value_for_bool(False, self.parameters['allow_flexgroups'])
        attach_object_store = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-attach', **store_obj)
        try:
            self.server.invoke_successfully(attach_object_store, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error attaching object store %s to aggregate %s: %s' % (self.parameters['object_store_name'], self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def add_disks(self, count=0, disks=None, mirror_disks=None, disk_size=0, disk_size_with_unit=None):
        """
        Add additional disks to aggregate.
        :return: None
        """
        if self.use_rest:
            return self.add_disks_rest(count, disks, mirror_disks, disk_size, disk_size_with_unit)
        options = {'aggregate': self.parameters['name']}
        if count:
            options['disk-count'] = str(count)
        if disks and self.parameters.get('ignore_pool_checks'):
            options['ignore-pool-checks'] = str(self.parameters['ignore_pool_checks'])
        if disk_size:
            options['disk-size'] = str(disk_size)
        if disk_size_with_unit:
            options['disk-size-with-unit'] = disk_size_with_unit
        if self.parameters.get('disk_class'):
            options['disk-class'] = self.parameters['disk_class']
        if self.parameters.get('disk_type'):
            options['disk-type'] = self.parameters['disk_type']
        aggr_add = netapp_utils.zapi.NaElement.create_node_with_children('aggr-add', **options)
        if disks:
            aggr_add.add_child_elem(self.get_disks_or_mirror_disks_object('disks', disks))
        if mirror_disks:
            aggr_add.add_child_elem(self.get_disks_or_mirror_disks_object('mirror-disks', mirror_disks))
        try:
            self.server.invoke_successfully(aggr_add, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error adding additional disks to aggregate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def map_plex_to_primary_and_mirror(self, plex_disks, disks, mirror_disks):
        """
        we have N plexes, and disks, and maybe mirror_disks
        we're trying to find which plex is used for disks, and which one, if applicable, for mirror_disks
        :return: a tuple with the names of the two plexes (disks_plex, mirror_disks_plex)
        the second one can be None
        """
        disks_plex = None
        mirror_disks_plex = None
        error = ''
        for plex in plex_disks:
            common = set(plex_disks[plex]).intersection(set(disks))
            if common:
                if disks_plex is None:
                    disks_plex = plex
                else:
                    error = 'found overlapping plexes: %s and %s' % (disks_plex, plex)
            if mirror_disks is not None:
                common = set(plex_disks[plex]).intersection(set(mirror_disks))
                if common:
                    if mirror_disks_plex is None:
                        mirror_disks_plex = plex
                    else:
                        error = 'found overlapping mirror plexes: %s and %s' % (mirror_disks_plex, plex)
        if not error:
            if disks_plex is None:
                error = 'cannot match disks with current aggregate disks'
            if mirror_disks is not None and mirror_disks_plex is None:
                if error:
                    error += ', and '
                error += 'cannot match mirror_disks with current aggregate disks'
        if error:
            self.module.fail_json(msg='Error mapping disks for aggregate %s: %s.  Found: %s' % (self.parameters['name'], error, str(plex_disks)))
        return (disks_plex, mirror_disks_plex)

    def get_disks_to_add(self, aggr_name, disks, mirror_disks):
        """
        Get list of disks used by the aggregate, as primary and mirror.
        Report error if:
          the plexes in use cannot be matched with user inputs (we expect some overlap)
          the user request requires some disks to be removed (not supported)
        : return: a tuple of two lists of disks: disks_to_add, mirror_disks_to_add
        """
        disks_in_use = self.get_aggr_disks(aggr_name)
        plex_disks = {}
        for disk_name, plex_name in disks_in_use:
            plex_disks.setdefault(plex_name, []).append(disk_name)
        disks_plex, mirror_disks_plex = self.map_plex_to_primary_and_mirror(plex_disks, disks, mirror_disks)
        disks_to_remove = [disk for disk in plex_disks[disks_plex] if disk not in disks]
        if mirror_disks_plex:
            disks_to_remove.extend([disk for disk in plex_disks[mirror_disks_plex] if disk not in mirror_disks])
        if disks_to_remove:
            error = 'these disks cannot be removed: %s' % str(disks_to_remove)
            self.module.fail_json(msg='Error removing disks is not supported.  Aggregate %s: %s.  In use: %s' % (aggr_name, error, str(plex_disks)))
        disks_to_add = [disk for disk in disks if disk not in plex_disks[disks_plex]]
        mirror_disks_to_add = []
        if mirror_disks_plex:
            mirror_disks_to_add = [disk for disk in mirror_disks if disk not in plex_disks[mirror_disks_plex]]
        if mirror_disks_to_add and (not disks_to_add):
            self.module.fail_json(msg='Error cannot add mirror disks %s without adding disks for aggregate %s.  In use: %s' % (str(mirror_disks_to_add), aggr_name, str(plex_disks)))
        if disks_to_add or mirror_disks_to_add:
            self.na_helper.changed = True
        return (disks_to_add, mirror_disks_to_add)

    def set_disk_count(self, current, modify):
        if modify.get('disk_count'):
            if int(modify['disk_count']) < int(current['disk_count']):
                self.module.fail_json(msg='Error: specified disk_count is less than current disk_count. Only adding disks is allowed.')
            else:
                modify['disk_count'] = modify['disk_count'] - current['disk_count']

    def get_aggr_actions(self):
        aggr_name = self.parameters.get('name')
        rename, cd_action, modify = (None, None, {})
        current = self.get_aggr()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and self.parameters.get('from_name'):
            old_aggregate = self.get_aggr(self.parameters['from_name'])
            rename = self.na_helper.is_rename_action(old_aggregate, current)
            if rename is None:
                self.module.fail_json(msg='Error renaming aggregate %s: no aggregate with from_name %s.' % (self.parameters['name'], self.parameters['from_name']))
            if rename:
                current = old_aggregate
                aggr_name = self.parameters['from_name']
                cd_action = None
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if 'encryption' in modify and (not self.use_rest):
                self.module.fail_json(msg='Error: modifying encryption is not supported with ZAPI.')
            if 'snaplock_type' in modify:
                self.module.fail_json(msg='Error: snaplock_type is not modifiable.  Cannot change to: %s.' % modify['snaplock_type'])
            if self.parameters.get('disks'):
                modify['disks_to_add'], modify['mirror_disks_to_add'] = self.get_disks_to_add(aggr_name, self.parameters['disks'], self.parameters.get('mirror_disks'))
            self.set_disk_count(current, modify)
        return (current, cd_action, rename, modify)

    def get_object_store_action(self, current, rename):
        object_store_cd_action = None
        if self.parameters.get('object_store_name'):
            aggr_name = self.parameters['from_name'] if rename else self.parameters['name']
            object_store_current = self.get_object_store(aggr_name) if current else None
            object_store_cd_action = self.na_helper.get_cd_action(object_store_current, self.parameters.get('object_store_name'))
            if object_store_cd_action is None and object_store_current is not None and (object_store_current['object_store_name'] != self.parameters.get('object_store_name')):
                self.module.fail_json(msg='Error: object store %s is already associated with aggregate %s.' % (object_store_current['object_store_name'], aggr_name))
        return object_store_cd_action

    def get_aggr_rest(self, name):
        if not name:
            return None
        api = 'storage/aggregates'
        query = {'name': name}
        fields = 'uuid,state,block_storage.primary.disk_count,data_encryption,snaplock_type'
        if 'tags' in self.parameters:
            fields += ',_tags'
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg='Error: failed to get aggregate %s: %s' % (name, error))
        if record:
            return {'tags': record.get('_tags', []), 'disk_count': self.na_helper.safe_get(record, ['block_storage', 'primary', 'disk_count']), 'encryption': self.na_helper.safe_get(record, ['data_encryption', 'software_encryption_enabled']), 'service_state': record['state'], 'snaplock_type': record['snaplock_type'], 'uuid': record['uuid']}
        return None

    def get_multiplier(self, unit):
        if not unit:
            return 1
        try:
            return netapp_utils.POW2_BYTE_MAP[unit[0].lower()]
        except KeyError:
            self.module.fail_json(msg='Error: unexpected unit in disk_size_with_unit: %s' % self.parameters['disk_size_with_unit'])

    def get_disk_size(self):
        if 'disk_size' in self.parameters:
            return self.parameters['disk_size'] * 4 * 1024
        if 'disk_size_with_unit' in self.parameters:
            match = re.match('([\\d.]+)(.*)', self.parameters['disk_size_with_unit'])
            if match:
                size, unit = match.groups()
                mul = self.get_multiplier(unit)
                return int(float(size) * mul)
            self.module.fail_json(msg='Error: unexpected value in disk_size_with_unit: %s' % self.parameters['disk_size_with_unit'])
        return None

    def create_aggr_rest(self):
        api = 'storage/aggregates'
        disk_size = self.get_disk_size()
        query = {'return_records': 'true'}
        if disk_size:
            query['disk_size'] = disk_size
        body = {'name': self.parameters['name']} if 'name' in self.parameters else {}
        block_storage = {}
        primary = {}
        if self.parameters.get('nodes'):
            body['node.name'] = self.parameters['nodes'][0]
        if self.parameters.get('disk_class'):
            primary['disk_class'] = self.parameters['disk_class']
        if self.parameters.get('disk_count'):
            primary['disk_count'] = self.parameters['disk_count']
        if self.parameters.get('raid_size'):
            primary['raid_size'] = self.parameters['raid_size']
        if self.parameters.get('raid_type'):
            primary['raid_type'] = self.parameters['raid_type']
        if primary:
            block_storage['primary'] = primary
        mirror = {}
        if self.parameters.get('is_mirrored'):
            mirror['enabled'] = self.parameters['is_mirrored']
        if mirror:
            block_storage['mirror'] = mirror
        if block_storage:
            body['block_storage'] = block_storage
        if self.parameters.get('encryption'):
            body['data_encryption'] = {'software_encryption_enabled': True}
        if self.parameters.get('snaplock_type'):
            body['snaplock_type'] = self.parameters['snaplock_type']
        if self.parameters.get('tags') is not None:
            body['_tags'] = self.parameters['tags']
        response, error = rest_generic.post_async(self.rest_api, api, body or None, query, job_timeout=self.parameters['time_out'])
        if error:
            self.module.fail_json(msg='Error: failed to create aggregate: %s' % error)
        if response:
            record, error = rrh.check_for_0_or_1_records(api, response, error, query)
            if not error and record and ('uuid' not in record):
                error = 'uuid key not present in %s:' % record
            if error:
                self.module.fail_json(msg='Error: failed to parse create aggregate response: %s' % error)
            if record:
                self.uuid = record['uuid']

    def delete_aggr_rest(self):
        api = 'storage/aggregates'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid)
        if error:
            self.module.fail_json(msg='Error: failed to delete aggregate: %s' % error)

    def patch_aggr_rest(self, action, body, query=None):
        api = 'storage/aggregates'
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body, query)
        if error:
            self.module.fail_json(msg='Error: failed to %s aggregate: %s' % (action, error))

    def add_disks_rest(self, count=0, disks=None, mirror_disks=None, disk_size=0, disk_size_with_unit=None):
        """
        Add additional disks to aggregate.
        :return: None
        """
        if disks or mirror_disks:
            self.module.fail_json(msg='Error: disks or mirror disks are mot supported with rest: %s, %s.' % (disks, mirror_disks))
        if self.parameters.get('disk_class'):
            self.module.warn('disk_class is ignored when adding disks to an exiting aggregate')
        primary = {'disk_count': self.parameters['disk_count']} if count else None
        body = {'block_storage': {'primary': primary}} if primary else None
        if body:
            disk_size = self.get_disk_size()
            query = {'disk_size': disk_size} if disk_size else None
            self.patch_aggr_rest('increase disk count for', body, query)

    def rename_aggr_rest(self):
        body = {'name': self.parameters['name']}
        self.patch_aggr_rest('rename', body)

    def get_object_store_rest(self):
        """TODO: support mirror in addition to primary"""
        api = 'storage/aggregates/%s/cloud-stores' % self.uuid
        record, error = rest_generic.get_one_record(self.rest_api, api, query={'primary': True})
        if error:
            self.module.fail_json(msg='Error: failed to get cloud stores for aggregate: %s' % error)
        return record

    def get_cloud_target_uuid_rest(self):
        api = 'cloud/targets'
        query = {'name': self.parameters['object_store_name']}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error or not record:
            self.module.fail_json(msg='Error: failed to find cloud store with name %s: %s' % (self.parameters['object_store_name'], error))
        return record['uuid']

    def attach_object_store_to_aggr_rest(self):
        """TODO: support mirror in addition to primary"""
        if self.uuid is None:
            error = 'aggregate UUID is not set.'
            self.module.fail_json(msg='Error: cannot attach cloud store with name %s: %s' % (self.parameters['object_store_name'], error))
        body = {'target': {'uuid': self.get_cloud_target_uuid_rest()}}
        api = 'storage/aggregates/%s/cloud-stores' % self.uuid
        query = None
        if 'allow_flexgroups' in self.parameters:
            query = {'allow_flexgroups': 'true' if self.parameters['allow_flexgroups'] else 'false'}
        record, error = rest_generic.post_async(self.rest_api, api, body, query)
        if error:
            self.module.fail_json(msg='Error: failed to attach cloud store with name %s: %s' % (self.parameters['object_store_name'], error))
        return record

    def validate_expensive_options(self, cd_action, modify):
        if cd_action == 'create' or (modify and 'disk_count' in modify):
            self.get_disk_size()

    def apply(self):
        """
        Apply action to the aggregate
        :return: None
        """
        current, cd_action, rename, modify = self.get_aggr_actions()
        if current:
            self.uuid = current.get('uuid')
        object_store_cd_action = self.get_object_store_action(current, rename)
        if self.na_helper.changed and self.module.check_mode:
            self.validate_expensive_options(cd_action, modify)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_aggr()
                if self.parameters.get('service_state') == 'offline':
                    self.modify_aggr({'service_state': 'offline'})
            elif cd_action == 'delete':
                self.delete_aggr()
            else:
                if rename:
                    self.rename_aggregate()
                if modify:
                    self.modify_aggr(modify)
            if object_store_cd_action == 'create':
                self.attach_object_store_to_aggr()
        if rename:
            modify['name'] = self.parameters['name']
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)