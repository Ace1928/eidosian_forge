from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
class ElementSWSnapShotSchedule(object):
    """
    Contains methods to parse arguments,
    derive details of ElementSW objects
    and send requests to ElementSW via
    the ElementSW SDK
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check paramenters and ensure SDK is installed
        """
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), schedule_type=dict(required=False, choices=['DaysOfWeekFrequency', 'DaysOfMonthFrequency', 'TimeIntervalFrequency']), time_interval_days=dict(required=False, type='int'), time_interval_hours=dict(required=False, type='int'), time_interval_minutes=dict(required=False, type='int'), days_of_week_weekdays=dict(required=False, type='list', elements='str'), days_of_week_hours=dict(required=False, type='int'), days_of_week_minutes=dict(required=False, type='int'), days_of_month_monthdays=dict(required=False, type='list', elements='int'), days_of_month_hours=dict(required=False, type='int'), days_of_month_minutes=dict(required=False, type='int'), paused=dict(required=False, type='bool'), recurring=dict(required=False, type='bool'), starting_date=dict(required=False, type='str'), snapshot_name=dict(required=False, type='str'), volumes=dict(required=False, type='list', elements='str'), account_id=dict(required=False, type='str'), retention=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['account_id', 'volumes', 'schedule_type']), ('schedule_type', 'DaysOfMonthFrequency', ['days_of_month_monthdays']), ('schedule_type', 'DaysOfWeekFrequency', ['days_of_week_weekdays'])], supports_check_mode=True)
        param = self.module.params
        self.state = param['state']
        self.name = param['name']
        self.schedule_type = param['schedule_type']
        self.days_of_week_weekdays = param['days_of_week_weekdays']
        self.days_of_week_hours = param['days_of_week_hours']
        self.days_of_week_minutes = param['days_of_week_minutes']
        self.days_of_month_monthdays = param['days_of_month_monthdays']
        self.days_of_month_hours = param['days_of_month_hours']
        self.days_of_month_minutes = param['days_of_month_minutes']
        self.time_interval_days = param['time_interval_days']
        self.time_interval_hours = param['time_interval_hours']
        self.time_interval_minutes = param['time_interval_minutes']
        self.paused = param['paused']
        self.recurring = param['recurring']
        if self.schedule_type == 'DaysOfWeekFrequency':
            if self.days_of_week_weekdays is not None:
                self.weekdays = []
                for day in self.days_of_week_weekdays:
                    if str(day).isdigit():
                        self.weekdays.append(Weekday.from_id(int(day)))
                    else:
                        self.weekdays.append(Weekday.from_name(day.capitalize()))
        if self.state == 'present' and self.schedule_type is None:
            self.module.fail_json(msg='Please provide required parameter: schedule_type')
        if self.state == 'absent' and self.name is None:
            self.module.fail_json(msg='Please provide required parameter: name')
        self.starting_date = param['starting_date']
        self.snapshot_name = param['snapshot_name']
        self.volumes = param['volumes']
        self.account_id = param['account_id']
        self.retention = param['retention']
        self.create_schedule_result = None
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the ElementSW Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)
            self.elementsw_helper = NaElementSWModule(self.sfe)

    def get_schedule(self):
        try:
            schedule_list = self.sfe.list_schedules()
        except ApiServerError:
            return None
        for schedule in schedule_list.schedules:
            if schedule.to_be_deleted:
                continue
            if str(schedule.schedule_id) == self.name:
                self.name = schedule.name
                return schedule
            elif schedule.name == self.name:
                return schedule
        return None

    def get_account_id(self):
        try:
            account_id = self.elementsw_helper.account_exists(self.account_id)
            return account_id
        except ApiServerError:
            return None

    def get_volume_id(self):
        volume_ids = []
        for volume in self.volumes:
            volume_id = self.elementsw_helper.volume_exists(volume.strip(), self.account_id)
            if volume_id:
                volume_ids.append(volume_id)
            else:
                self.module.fail_json(msg='Specified volume %s does not exist' % volume)
        return volume_ids

    def get_frequency(self):
        frequency = None
        if self.schedule_type is not None and self.schedule_type == 'DaysOfWeekFrequency':
            if self.weekdays is not None:
                params = dict(weekdays=self.weekdays)
                if self.days_of_week_hours is not None:
                    params['hours'] = self.days_of_week_hours
                if self.days_of_week_minutes is not None:
                    params['minutes'] = self.days_of_week_minutes
                frequency = DaysOfWeekFrequency(**params)
        elif self.schedule_type is not None and self.schedule_type == 'DaysOfMonthFrequency':
            if self.days_of_month_monthdays is not None:
                params = dict(monthdays=self.days_of_month_monthdays)
                if self.days_of_month_hours is not None:
                    params['hours'] = self.days_of_month_hours
                if self.days_of_month_minutes is not None:
                    params['minutes'] = self.days_of_month_minutes
                frequency = DaysOfMonthFrequency(**params)
        elif self.schedule_type is not None and self.schedule_type == 'TimeIntervalFrequency':
            params = dict()
            if self.time_interval_days is not None:
                params['days'] = self.time_interval_days
            if self.time_interval_hours is not None:
                params['hours'] = self.time_interval_hours
            if self.time_interval_minutes is not None:
                params['minutes'] = self.time_interval_minutes
            if not params or sum(params.values()) == 0:
                self.module.fail_json(msg='Specify at least one non zero value with TimeIntervalFrequency.')
            frequency = TimeIntervalFrequency(**params)
        return frequency

    def is_same_schedule_type(self, schedule_detail):
        if str(schedule_detail.frequency).split('(', maxsplit=1)[0] == self.schedule_type:
            return True
        else:
            return False

    def create_schedule(self):
        try:
            frequency = self.get_frequency()
            if frequency is None:
                self.module.fail_json(msg='Failed to create schedule frequency object - type %s parameters' % self.schedule_type)
            name = self.name
            schedule_info = ScheduleInfo(volume_ids=self.volumes, snapshot_name=self.snapshot_name, retention=self.retention)
            if HAS_SF_SDK_1_7:
                sched = Schedule(frequency, name, schedule_info)
            else:
                sched = Schedule(schedule_info, name, frequency)
            sched.paused = self.paused
            sched.recurring = self.recurring
            sched.starting_date = self.starting_date
            self.create_schedule_result = self.sfe.create_schedule(sched)
        except (ApiServerError, ApiConnectionError) as exc:
            self.module.fail_json(msg='Error creating schedule %s: %s' % (self.name, to_native(exc)), exception=traceback.format_exc())

    def delete_schedule(self, schedule_id):
        try:
            get_schedule_result = self.sfe.get_schedule(schedule_id=schedule_id)
            sched = get_schedule_result.schedule
            sched.to_be_deleted = True
            self.sfe.modify_schedule(schedule=sched)
        except (ApiServerError, ApiConnectionError) as exc:
            self.module.fail_json(msg='Error deleting schedule %s: %s' % (self.name, to_native(exc)), exception=traceback.format_exc())

    def update_schedule(self, schedule_id):
        try:
            get_schedule_result = self.sfe.get_schedule(schedule_id=schedule_id)
            sched = get_schedule_result.schedule
            sched.frequency = self.get_frequency()
            if sched.frequency is None:
                self.module.fail_json(msg='Failed to create schedule frequency object - type %s parameters' % self.schedule_type)
            if self.volumes is not None and len(self.volumes) > 0:
                sched.schedule_info.volume_ids = self.volumes
            if self.retention is not None:
                sched.schedule_info.retention = self.retention
            if self.snapshot_name is not None:
                sched.schedule_info.snapshot_name = self.snapshot_name
            if self.paused is not None:
                sched.paused = self.paused
            if self.recurring is not None:
                sched.recurring = self.recurring
            if self.starting_date is not None:
                sched.starting_date = self.starting_date
            self.sfe.modify_schedule(schedule=sched)
        except (ApiServerError, ApiConnectionError) as exc:
            self.module.fail_json(msg='Error updating schedule %s: %s' % (self.name, to_native(exc)), exception=traceback.format_exc())

    def apply(self):
        changed = False
        update_schedule = False
        if self.account_id is not None:
            self.account_id = self.get_account_id()
        if self.state == 'present' and self.volumes is not None:
            if self.account_id:
                self.volumes = self.get_volume_id()
            else:
                self.module.fail_json(msg='Specified account id does not exist')
        schedule_detail = self.get_schedule()
        if schedule_detail is None and self.state == 'present':
            if len(self.volumes) > 0:
                changed = True
            else:
                self.module.fail_json(msg='Specified volumes not on cluster')
        elif schedule_detail is not None:
            if self.state == 'absent':
                changed = True
            else:
                if self.retention is not None and schedule_detail.schedule_info.retention != self.retention:
                    update_schedule = True
                    changed = True
                elif self.snapshot_name is not None and schedule_detail.schedule_info.snapshot_name != self.snapshot_name:
                    update_schedule = True
                    changed = True
                elif self.paused is not None and schedule_detail.paused != self.paused:
                    update_schedule = True
                    changed = True
                elif self.recurring is not None and schedule_detail.recurring != self.recurring:
                    update_schedule = True
                    changed = True
                elif self.starting_date is not None and schedule_detail.starting_date != self.starting_date:
                    update_schedule = True
                    changed = True
                elif self.volumes is not None and len(self.volumes) > 0:
                    for volume_id in schedule_detail.schedule_info.volume_ids:
                        if volume_id not in self.volumes:
                            update_schedule = True
                            changed = True
                temp_frequency = self.get_frequency()
                if temp_frequency is not None:
                    if self.is_same_schedule_type(schedule_detail):
                        if self.schedule_type == 'TimeIntervalFrequency':
                            if schedule_detail.frequency.days != temp_frequency.days or schedule_detail.frequency.hours != temp_frequency.hours or schedule_detail.frequency.minutes != temp_frequency.minutes:
                                update_schedule = True
                                changed = True
                        elif self.schedule_type == 'DaysOfMonthFrequency':
                            if len(schedule_detail.frequency.monthdays) != len(temp_frequency.monthdays) or schedule_detail.frequency.hours != temp_frequency.hours or schedule_detail.frequency.minutes != temp_frequency.minutes:
                                update_schedule = True
                                changed = True
                            elif len(schedule_detail.frequency.monthdays) == len(temp_frequency.monthdays):
                                actual_frequency_monthday = schedule_detail.frequency.monthdays
                                temp_frequency_monthday = temp_frequency.monthdays
                                for monthday in actual_frequency_monthday:
                                    if monthday not in temp_frequency_monthday:
                                        update_schedule = True
                                        changed = True
                        elif self.schedule_type == 'DaysOfWeekFrequency':
                            if len(schedule_detail.frequency.weekdays) != len(temp_frequency.weekdays) or schedule_detail.frequency.hours != temp_frequency.hours or schedule_detail.frequency.minutes != temp_frequency.minutes:
                                update_schedule = True
                                changed = True
                            elif len(schedule_detail.frequency.weekdays) == len(temp_frequency.weekdays):
                                actual_frequency_weekdays = schedule_detail.frequency.weekdays
                                temp_frequency_weekdays = temp_frequency.weekdays
                                if len([actual_weekday for actual_weekday, temp_weekday in zip(actual_frequency_weekdays, temp_frequency_weekdays) if actual_weekday != temp_weekday]) != 0:
                                    update_schedule = True
                                    changed = True
                    else:
                        update_schedule = True
                        changed = True
                else:
                    self.module.fail_json(msg='Failed to create schedule frequency object - type %s parameters' % self.schedule_type)
        result_message = ' '
        if changed:
            if self.module.check_mode:
                result_message = 'Check mode, skipping changes'
            elif self.state == 'present':
                if update_schedule:
                    self.update_schedule(schedule_detail.schedule_id)
                    result_message = 'Snapshot Schedule modified'
                else:
                    self.create_schedule()
                    result_message = 'Snapshot Schedule created'
            elif self.state == 'absent':
                self.delete_schedule(schedule_detail.schedule_id)
                result_message = 'Snapshot Schedule deleted'
        self.module.exit_json(changed=changed, msg=result_message)