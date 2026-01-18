from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapSnapshotPolicy(object):
    """
    Creates and deletes a Snapshot Policy
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), enabled=dict(required=False, type='bool'), count=dict(required=False, type='list', elements='int'), comment=dict(required=False, type='str'), schedule=dict(required=False, type='list', elements='str'), prefix=dict(required=False, type='list', elements='str'), snapmirror_label=dict(required=False, type='list', elements='str'), vserver=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['enabled', 'count', 'schedule'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if self.use_rest and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0)):
            msg = 'REST requires ONTAP 9.8 or later for snapshot schedules.'
            self.use_rest = self.na_helper.fall_back_to_zapi(self.module, msg, self.parameters)
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg='the python NetApp-Lib module is required')
            if 'vserver' in self.parameters:
                self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
            else:
                self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def safe_strip(self, option):
        """ strip the given string """
        return option.strip() if option is not None else None

    def get_snapshot_policy(self):
        """
        Checks to see if a snapshot policy exists or not
        :return: Return policy details if a snapshot policy exists, None if it doesn't
        """
        snapshot_obj = netapp_utils.zapi.NaElement('snapshot-policy-get-iter')
        query = netapp_utils.zapi.NaElement('query')
        snapshot_info_obj = netapp_utils.zapi.NaElement('snapshot-policy-info')
        snapshot_info_obj.add_new_child('policy', self.parameters['name'])
        if 'vserver' in self.parameters:
            snapshot_info_obj.add_new_child('vserver-name', self.parameters['vserver'])
        query.add_child_elem(snapshot_info_obj)
        snapshot_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(snapshot_obj, True)
            if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
                snapshot_policy = result.get_child_by_name('attributes-list').get_child_by_name('snapshot-policy-info')
                current = {'name': snapshot_policy.get_child_content('policy')}
                current['vserver'] = snapshot_policy.get_child_content('vserver-name')
                current['enabled'] = snapshot_policy.get_child_content('enabled').lower() != 'false'
                current['comment'] = snapshot_policy.get_child_content('comment') or ''
                current['schedule'], current['count'], current['snapmirror_label'], current['prefix'] = ([], [], [], [])
                if snapshot_policy.get_child_by_name('snapshot-policy-schedules'):
                    for schedule in snapshot_policy['snapshot-policy-schedules'].get_children():
                        current['schedule'].append(schedule.get_child_content('schedule'))
                        current['count'].append(int(schedule.get_child_content('count')))
                        snapmirror_label = schedule.get_child_content('snapmirror-label')
                        if snapmirror_label is None or snapmirror_label == '-':
                            snapmirror_label = ''
                        current['snapmirror_label'].append(snapmirror_label)
                        prefix = schedule.get_child_content('prefix')
                        if prefix is None or prefix == '-':
                            prefix = ''
                        current['prefix'].append(prefix)
                return current
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        return None

    def validate_parameters(self):
        """
        Validate if each schedule has a count associated
        :return: None
        """
        if 'count' not in self.parameters or 'schedule' not in self.parameters or len(self.parameters['count']) > 5 or (len(self.parameters['schedule']) > 5) or (len(self.parameters['count']) < 1) or (len(self.parameters['schedule']) < 1) or (len(self.parameters['count']) != len(self.parameters['schedule'])):
            self.module.fail_json(msg='Error: A Snapshot policy must have at least 1 schedule and can have up to a maximum of 5 schedules, with a count representing the maximum number of Snapshot copies for each schedule')
        if 'snapmirror_label' in self.parameters and len(self.parameters['snapmirror_label']) != len(self.parameters['schedule']):
            self.module.fail_json(msg='Error: Each Snapshot Policy schedule must have an accompanying SnapMirror Label')
        if 'prefix' in self.parameters and len(self.parameters['prefix']) != len(self.parameters['schedule']):
            self.module.fail_json(msg='Error: Each Snapshot Policy schedule must have an accompanying prefix')

    def modify_snapshot_policy(self, current):
        """
        Modifies an existing snapshot policy
        """
        options = {'policy': self.parameters['name']}
        modify = False
        if 'enabled' in self.parameters and self.parameters['enabled'] != current['enabled']:
            options['enabled'] = str(self.parameters['enabled'])
            modify = True
        if 'comment' in self.parameters and self.parameters['comment'] != current['comment']:
            options['comment'] = self.parameters['comment']
            modify = True
        if modify:
            snapshot_obj = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-policy-modify', **options)
            try:
                self.server.invoke_successfully(snapshot_obj, True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error modifying snapshot policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_snapshot_policy_schedules(self, current):
        """
        Modify existing schedules in snapshot policy
        :return: None
        """
        self.validate_parameters()
        delete_schedules, modify_schedules, add_schedules = ([], [], [])
        if 'snapmirror_label' in self.parameters:
            snapmirror_labels = self.parameters['snapmirror_label']
        else:
            snapmirror_labels = [None] * len(self.parameters['schedule'])
        for schedule in current['schedule']:
            schedule = self.safe_strip(schedule)
            if schedule not in [item.strip() for item in self.parameters['schedule']]:
                options = {'policy': current['name'], 'schedule': schedule}
                delete_schedules.append(options)
        for schedule, count, snapmirror_label in zip(self.parameters['schedule'], self.parameters['count'], snapmirror_labels):
            schedule = self.safe_strip(schedule)
            snapmirror_label = self.safe_strip(snapmirror_label)
            options = {'policy': current['name'], 'schedule': schedule}
            if schedule in current['schedule']:
                modify = False
                schedule_index = current['schedule'].index(schedule)
                if count != current['count'][schedule_index]:
                    options['new-count'] = str(count)
                    modify = True
                if snapmirror_label is not None and snapmirror_label != current['snapmirror_label'][schedule_index]:
                    options['new-snapmirror-label'] = snapmirror_label
                    modify = True
                if modify:
                    modify_schedules.append(options)
            else:
                options['count'] = str(count)
                if snapmirror_label is not None and snapmirror_label != '':
                    options['snapmirror-label'] = snapmirror_label
                add_schedules.append(options)
        while len(delete_schedules) > 1:
            options = delete_schedules.pop()
            self.modify_snapshot_policy_schedule(options, 'snapshot-policy-remove-schedule')
        while modify_schedules:
            options = modify_schedules.pop()
            self.modify_snapshot_policy_schedule(options, 'snapshot-policy-modify-schedule')
        if add_schedules:
            options = add_schedules.pop()
            self.modify_snapshot_policy_schedule(options, 'snapshot-policy-add-schedule')
        while delete_schedules:
            options = delete_schedules.pop()
            self.modify_snapshot_policy_schedule(options, 'snapshot-policy-remove-schedule')
        while add_schedules:
            options = add_schedules.pop()
            self.modify_snapshot_policy_schedule(options, 'snapshot-policy-add-schedule')

    def modify_snapshot_policy_schedule(self, options, zapi):
        """
        Add, modify or remove a schedule to/from a snapshot policy
        """
        snapshot_obj = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
        try:
            self.server.invoke_successfully(snapshot_obj, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying snapshot policy schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def create_snapshot_policy(self):
        """
        Creates a new snapshot policy
        """
        self.validate_parameters()
        options = {'policy': self.parameters['name'], 'enabled': str(self.parameters['enabled'])}
        if 'snapmirror_label' in self.parameters:
            snapmirror_labels = self.parameters['snapmirror_label']
        else:
            snapmirror_labels = [None] * len(self.parameters['schedule'])
        if 'prefix' in self.parameters:
            prefixes = self.parameters['prefix']
        else:
            prefixes = [None] * len(self.parameters['schedule'])
        positions = [str(i) for i in range(1, len(self.parameters['schedule']) + 1)]
        for schedule, prefix, count, snapmirror_label, position in zip(self.parameters['schedule'], prefixes, self.parameters['count'], snapmirror_labels, positions):
            schedule = self.safe_strip(schedule)
            options['count' + position] = str(count)
            options['schedule' + position] = schedule
            snapmirror_label = self.safe_strip(snapmirror_label)
            if snapmirror_label:
                options['snapmirror-label' + position] = snapmirror_label
            prefix = self.safe_strip(prefix)
            if prefix:
                options['prefix' + position] = prefix
        snapshot_obj = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-policy-create', **options)
        if self.parameters.get('comment'):
            snapshot_obj.add_new_child('comment', self.parameters['comment'])
        try:
            self.server.invoke_successfully(snapshot_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating snapshot policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_snapshot_policy(self):
        """
        Deletes an existing snapshot policy
        """
        snapshot_obj = netapp_utils.zapi.NaElement('snapshot-policy-delete')
        snapshot_obj.add_new_child('policy', self.parameters['name'])
        try:
            self.server.invoke_successfully(snapshot_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting snapshot policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def asup_log_for_cserver(self, event_name):
        """
        Fetch admin vserver for the given cluster
        Create and Autosupport log event with the given module name
        :param event_name: Name of the event log
        :return: None
        """
        if 'vserver' in self.parameters:
            netapp_utils.ems_log_event(event_name, self.server)
        else:
            results = netapp_utils.get_cserver(self.server)
            cserver = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=results)
            netapp_utils.ems_log_event(event_name, cserver)

    def get_snapshot_schedule_rest(self, current):
        """
        get details of the snapshot schedule with rest API.
        """
        query = {'snapshot_policy.name': current['name']}
        api = 'storage/snapshot-policies/%s/schedules' % current['uuid']
        fields = 'schedule.name,schedule.uuid,snapmirror_label,count,prefix'
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg='Error on fetching snapshot schedule: %s' % error)
        if records:
            scheduleRecords = {'counts': [], 'prefixes': [], 'schedule_names': [], 'schedule_uuids': [], 'snapmirror_labels': []}
            for item in records:
                scheduleRecords['counts'].append(item['count'])
                scheduleRecords['prefixes'].append(item['prefix'])
                scheduleRecords['schedule_names'].append(item['schedule']['name'])
                scheduleRecords['schedule_uuids'].append(item['schedule']['uuid'])
                scheduleRecords['snapmirror_labels'].append(item['snapmirror_label'])
            return scheduleRecords
        return None

    def get_snapshot_policy_rest(self):
        """
        get details of the snapshot policy with rest API.
        """
        if not self.use_rest:
            return self.get_snapshot_policy()
        query = {'name': self.parameters['name']}
        if self.parameters.get('vserver'):
            query['svm.name'] = self.parameters['vserver']
            query['scope'] = 'svm'
        else:
            query['scope'] = 'cluster'
        api = 'storage/snapshot-policies'
        fields = 'enabled,svm.uuid,comment,copies.snapmirror_label,copies.count,copies.prefix,copies.schedule.name,scope'
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg='Error on fetching snapshot policy: %s' % error)
        if record:
            current = {'enabled': record['enabled'], 'name': record['name'], 'uuid': record['uuid'], 'comment': record.get('comment', ''), 'count': [], 'prefix': [], 'schedule': [], 'snapmirror_label': []}
            if query['scope'] == 'svm':
                current['svm_name'] = record['svm']['name']
                current['svm_uuid'] = record['svm']['uuid']
            if record['copies']:
                for item in record['copies']:
                    current['count'].append(item['count'])
                    current['prefix'].append(item['prefix'])
                    current['schedule'].append(item['schedule']['name'])
                    current['snapmirror_label'].append(item['snapmirror_label'])
            return current
        return record

    def create_snapshot_policy_rest(self):
        """
        create snapshot policy with rest API.
        """
        if not self.use_rest:
            return self.create_snapshot_policy()
        body = {'name': self.parameters.get('name'), 'enabled': self.parameters.get('enabled'), 'copies': []}
        if self.parameters.get('vserver'):
            body['svm.name'] = self.parameters['vserver']
        if 'comment' in self.parameters:
            body['comment'] = self.parameters['comment']
        if 'snapmirror_label' in self.parameters:
            snapmirror_labels = self.parameters['snapmirror_label']
        else:
            snapmirror_labels = [None] * len(self.parameters['schedule'])
        if 'prefix' in self.parameters:
            prefixes = self.parameters['prefix']
        else:
            prefixes = [None] * len(self.parameters['schedule'])
        for schedule, prefix, count, snapmirror_label in zip(self.parameters['schedule'], prefixes, self.parameters['count'], snapmirror_labels):
            copy = {'schedule': {'name': self.safe_strip(schedule)}, 'count': count}
            snapmirror_label = self.safe_strip(snapmirror_label)
            if snapmirror_label:
                copy['snapmirror_label'] = snapmirror_label
            prefix = self.safe_strip(prefix)
            if prefix:
                copy['prefix'] = prefix
            body['copies'].append(copy)
        api = 'storage/snapshot-policies'
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error on creating snapshot policy: %s' % error)

    def delete_snapshot_policy_rest(self, current):
        """
        delete snapshot policy with rest API.
        """
        if not self.use_rest:
            return self.delete_snapshot_policy()
        api = 'storage/snapshot-policies'
        dummy, error = rest_generic.delete_async(self.rest_api, api, current['uuid'])
        if error is not None:
            self.module.fail_json(msg='Error on deleting snapshot policy: %s' % error)

    def modify_snapshot_policy_rest(self, modify, current=None):
        """
        Modify snapshot policy with rest API.
        """
        if not self.use_rest:
            return self.modify_snapshot_policy(current)
        api = 'storage/snapshot-policies'
        body = {}
        if 'enabled' in modify:
            body['enabled'] = modify['enabled']
        if 'comment' in modify:
            body['comment'] = modify['comment']
        if body:
            dummy, error = rest_generic.patch_async(self.rest_api, api, current['uuid'], body)
            if error is not None:
                self.module.fail_json(msg='Error on modifying snapshot policy: %s' % error)

    def modify_snapshot_policy_schedule_rest(self, modify, current):
        """
        Modify snapshot schedule with rest API.
        """
        if not self.use_rest:
            return self.modify_snapshot_policy_schedules(current)
        schedule_info = None
        api = 'storage/snapshot-policies/%s/schedules' % current['uuid']
        schedule_info = self.get_snapshot_schedule_rest(current)
        delete_schedules, modify_schedules, add_schedules = ([], [], [])
        retain_schedules_count = 0
        if 'snapmirror_label' in self.parameters:
            snapmirror_labels = self.parameters['snapmirror_label']
        else:
            snapmirror_labels = [None] * len(self.parameters['schedule'])
        if 'prefix' in self.parameters:
            prefixes = self.parameters['prefix']
        else:
            prefixes = [None] * len(self.parameters['schedule'])
        for schedule_name, schedule_uuid in zip(schedule_info['schedule_names'], schedule_info['schedule_uuids']):
            schedule_name = self.safe_strip(schedule_name)
            if schedule_name not in [item.strip() for item in self.parameters['schedule']]:
                delete_schedules.append(schedule_uuid)
            else:
                retain_schedules_count += 1
        for schedule_name, count, snapmirror_label, prefix in zip(self.parameters['schedule'], self.parameters['count'], snapmirror_labels, prefixes):
            schedule_name = self.safe_strip(schedule_name)
            if snapmirror_label:
                snapmirror_label = self.safe_strip(snapmirror_label)
            if prefix:
                prefix = self.safe_strip(prefix)
            body = {}
            if schedule_name in schedule_info['schedule_names']:
                modify = False
                schedule_index = schedule_info['schedule_names'].index(schedule_name)
                schedule_uuid = schedule_info['schedule_uuids'][schedule_index]
                if count != schedule_info['counts'][schedule_index]:
                    body['count'] = str(count)
                    modify = True
                if snapmirror_label is not None and snapmirror_label != schedule_info['snapmirror_labels'][schedule_index]:
                    body['snapmirror_label'] = snapmirror_label
                    modify = True
                if prefix is not None and prefix != schedule_info['prefixes'][schedule_index]:
                    body['prefix'] = prefix
                    modify = True
                if modify:
                    body['schedule_uuid'] = schedule_uuid
                    modify_schedules.append(body)
            else:
                body['schedule.name'] = schedule_name
                body['count'] = str(count)
                if snapmirror_label is not None and snapmirror_label != '':
                    body['snapmirror_label'] = snapmirror_label
                if prefix is not None and prefix != '':
                    body['prefix'] = prefix
                add_schedules.append(body)
        count = 0 if retain_schedules_count > 0 else 1
        while len(delete_schedules) > count:
            schedule_uuid = delete_schedules.pop()
            record, error = rest_generic.delete_async(self.rest_api, api, schedule_uuid)
            if error is not None:
                self.module.fail_json(msg='Error on deleting snapshot policy schedule: %s' % error)
        while modify_schedules:
            body = modify_schedules.pop()
            schedule_id = body.pop('schedule_uuid')
            record, error = rest_generic.patch_async(self.rest_api, api, schedule_id, body)
            if error is not None:
                self.module.fail_json(msg='Error on modifying snapshot policy schedule: %s' % error)
        if add_schedules:
            body = add_schedules.pop()
            record, error = rest_generic.post_async(self.rest_api, api, body)
            if error is not None:
                self.module.fail_json(msg='Error on adding snapshot policy schedule: %s' % error)
        while delete_schedules:
            schedule_uuid = delete_schedules.pop()
            record, error = rest_generic.delete_async(self.rest_api, api, schedule_uuid)
            if error is not None:
                self.module.fail_json(msg='Error on deleting snapshot policy schedule: %s' % error)
        while add_schedules:
            body = add_schedules.pop()
            record, error = rest_generic.post_async(self.rest_api, api, body)
            if error is not None:
                self.module.fail_json(msg='Error on adding snapshot policy schedule: %s' % error)

    def apply(self):
        """
        Check to see which play we should run
        """
        current = self.get_snapshot_policy_rest()
        modify = None
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.parameters['state'] == 'present':
            self.validate_parameters()
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_snapshot_policy_rest()
            elif cd_action == 'delete':
                self.delete_snapshot_policy_rest(current)
            if modify:
                self.modify_snapshot_policy_rest(modify, current)
                self.modify_snapshot_policy_schedule_rest(modify, current)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)