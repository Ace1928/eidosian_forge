from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPJob:
    """Class with job schedule cron methods"""

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), job_minutes=dict(required=False, type='list', elements='int'), job_months=dict(required=False, type='list', elements='int'), job_hours=dict(required=False, type='list', elements='int'), job_days_of_month=dict(required=False, type='list', elements='int'), job_days_of_week=dict(required=False, type='list', elements='int'), month_offset=dict(required=False, type='int', choices=[0, 1]), cluster=dict(required=False, type='str')))
        self.uuid = None
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        self.month_offset = self.parameters.get('month_offset')
        if self.month_offset is None:
            self.month_offset = 1 if self.use_rest else 0
        if self.month_offset == 1 and self.parameters.get('job_months') and (0 in self.parameters['job_months']):
            self.module.fail_json(msg='Error: 0 is not a valid value in months if month_offset is set to 1: %s' % self.parameters['job_months'])
        if self.use_rest:
            self.set_playbook_api_key_map()
        elif not netapp_utils.has_netapp_lib():
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        else:
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)
            self.set_playbook_zapi_key_map()

    def set_playbook_zapi_key_map(self):
        self.na_helper.zapi_string_keys = {'name': 'job-schedule-name', 'cluster': 'job-schedule-cluster'}
        self.na_helper.zapi_list_keys = {'job_minutes': ('job-schedule-cron-minute', 'cron-minute'), 'job_months': ('job-schedule-cron-month', 'cron-month'), 'job_hours': ('job-schedule-cron-hour', 'cron-hour'), 'job_days_of_month': ('job-schedule-cron-day', 'cron-day-of-month'), 'job_days_of_week': ('job-schedule-cron-day-of-week', 'cron-day-of-week')}

    def set_playbook_api_key_map(self):
        self.na_helper.params_to_rest_api_keys = {'job_minutes': 'minutes', 'job_months': 'months', 'job_hours': 'hours', 'job_days_of_month': 'days', 'job_days_of_week': 'weekdays'}

    def get_job_schedule_rest(self):
        """
        Return details about the job
        :param:
            name : Job name
        :return: Details about the Job. None if not found.
        :rtype: dict
        """
        query = {'name': self.parameters['name']}
        if self.parameters.get('cluster'):
            query['cluster'] = self.parameters['cluster']
        record, error = rest_generic.get_one_record(self.rest_api, 'cluster/schedules', query, 'uuid,cron')
        if error is not None:
            self.module.fail_json(msg='Error fetching job schedule: %s' % error)
        if record:
            self.uuid = record['uuid']
            job_details = {'name': record['name']}
            for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
                if rest_key in record['cron']:
                    job_details[param_key] = record['cron'][rest_key]
                else:
                    job_details[param_key] = [-1]
            if 'job_months' in job_details and self.month_offset == 0:
                job_details['job_months'] = [x - 1 if x > 0 else x for x in job_details['job_months']]
            if 'job_minutes' in job_details and len(job_details['job_minutes']) == 60:
                job_details['job_minutes'] = [-1]
            return job_details
        return None

    def get_job_schedule(self):
        """
        Return details about the job
        :param:
            name : Job name
        :return: Details about the Job. None if not found.
        :rtype: dict
        """
        if self.use_rest:
            return self.get_job_schedule_rest()
        job_get_iter = netapp_utils.zapi.NaElement('job-schedule-cron-get-iter')
        query = {'job-schedule-cron-info': {'job-schedule-name': self.parameters['name']}}
        if self.parameters.get('cluster'):
            query['job-schedule-cron-info']['job-schedule-cluster'] = self.parameters['cluster']
        job_get_iter.translate_struct({'query': query})
        try:
            result = self.server.invoke_successfully(job_get_iter, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        job_details = None
        if result.get_child_by_name('num-records') and int(result['num-records']) >= 1:
            job_info = result['attributes-list']['job-schedule-cron-info']
            job_details = {}
            for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
                job_details[item_key] = job_info[zapi_key]
            for item_key, zapi_key in self.na_helper.zapi_list_keys.items():
                parent, dummy = zapi_key
                job_details[item_key] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=job_info.get_child_by_name(parent))
                if item_key == 'job_months' and self.month_offset == 1:
                    job_details[item_key] = [int(x) + 1 if int(x) >= 0 else int(x) for x in job_details[item_key]]
                elif item_key == 'job_minutes' and len(job_details[item_key]) == 60:
                    job_details[item_key] = [-1]
                else:
                    job_details[item_key] = [int(x) for x in job_details[item_key]]
                if not job_details[item_key]:
                    job_details[item_key] = [-1]
        return job_details

    def add_job_details(self, na_element_object, values):
        """
        Add children node for create or modify NaElement object
        :param na_element_object: modify or create NaElement object
        :param values: dictionary of cron values to be added
        :return: None
        """
        for item_key, item_value in values.items():
            if item_key in self.na_helper.zapi_string_keys:
                zapi_key = self.na_helper.zapi_string_keys.get(item_key)
                na_element_object[zapi_key] = item_value
            elif item_key in self.na_helper.zapi_list_keys:
                parent_key, child_key = self.na_helper.zapi_list_keys.get(item_key)
                data = item_value
                if data:
                    if item_key == 'job_months' and self.month_offset == 1:
                        data = [str(x - 1) if x > 0 else str(x) for x in data]
                    else:
                        data = [str(x) for x in data]
                na_element_object.add_child_elem(self.na_helper.get_value_for_list(from_zapi=False, zapi_parent=parent_key, zapi_child=child_key, data=data))

    def create_job_schedule(self):
        """
        Creates a job schedule
        """
        if self.use_rest:
            cron = {}
            for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
                if self.parameters.get(param_key):
                    if len(self.parameters[param_key]) == 1 and self.parameters[param_key][0] == -1:
                        if rest_key == 'minutes':
                            cron[rest_key] = []
                    elif param_key == 'job_months' and self.month_offset == 0:
                        cron[rest_key] = [x + 1 if x >= 0 else x for x in self.parameters[param_key]]
                    else:
                        cron[rest_key] = self.parameters[param_key]
            params = {'name': self.parameters['name'], 'cron': cron}
            if self.parameters.get('cluster'):
                params['cluster'] = self.parameters['cluster']
            api = 'cluster/schedules'
            dummy, error = self.rest_api.post(api, params)
            if error is not None:
                self.module.fail_json(msg='Error creating job schedule: %s' % error)
        else:
            job_schedule_create = netapp_utils.zapi.NaElement('job-schedule-cron-create')
            self.add_job_details(job_schedule_create, self.parameters)
            try:
                self.server.invoke_successfully(job_schedule_create, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error creating job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_job_schedule(self):
        """
        Delete a job schedule
        """
        if self.use_rest:
            api = 'cluster/schedules/' + self.uuid
            dummy, error = self.rest_api.delete(api)
            if error is not None:
                self.module.fail_json(msg='Error deleting job schedule: %s' % error)
        else:
            job_schedule_delete = netapp_utils.zapi.NaElement('job-schedule-cron-destroy')
            self.add_job_details(job_schedule_delete, self.parameters)
            try:
                self.server.invoke_successfully(job_schedule_delete, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error deleting job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_job_schedule(self, modify, current):
        """
        modify a job schedule
        """

        def set_cron(param_key, rest_key, params, cron):
            if params[param_key] == [-1]:
                cron[rest_key] = []
            elif param_key == 'job_months' and self.month_offset == 0:
                cron[rest_key] = [x + 1 for x in params[param_key]]
            else:
                cron[rest_key] = params[param_key]
        if self.use_rest:
            cron = {}
            for param_key, rest_key in self.na_helper.params_to_rest_api_keys.items():
                if modify.get(param_key):
                    set_cron(param_key, rest_key, modify, cron)
                elif current.get(param_key):
                    set_cron(param_key, rest_key, current, cron)
            params = {'cron': cron}
            api = 'cluster/schedules/' + self.uuid
            dummy, error = self.rest_api.patch(api, params)
            if error is not None:
                self.module.fail_json(msg='Error modifying job schedule: %s' % error)
        else:
            job_schedule_modify = netapp_utils.zapi.NaElement.create_node_with_children('job-schedule-cron-modify', **{'job-schedule-name': self.parameters['name']})
            self.add_job_details(job_schedule_modify, modify)
            try:
                self.server.invoke_successfully(job_schedule_modify, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error modifying job schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        """
        Apply action to job-schedule
        """
        modify = None
        current = self.get_job_schedule()
        action = self.na_helper.get_cd_action(current, self.parameters)
        if action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if action == 'create' and self.parameters.get('job_minutes') is None:
            self.module.fail_json(msg='Error: missing required parameter job_minutes for create')
        if self.na_helper.changed and (not self.module.check_mode):
            if action == 'create':
                self.create_job_schedule()
            elif action == 'delete':
                self.delete_job_schedule()
            elif modify:
                self.modify_job_schedule(modify, current)
        result = netapp_utils.generate_result(self.na_helper.changed, action, modify)
        self.module.exit_json(**result)