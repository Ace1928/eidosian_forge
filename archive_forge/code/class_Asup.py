from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class Asup(object):
    DAYS_OPTIONS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=False, default='enabled', aliases=['asup', 'auto_support', 'autosupport'], choices=['enabled', 'disabled']), active=dict(type='bool', required=False, default=True), days=dict(type='list', required=False, aliases=['schedule_days', 'days_of_week'], choices=self.DAYS_OPTIONS), start=dict(type='int', required=False, default=0, aliases=['start_time']), end=dict(type='int', required=False, default=24, aliases=['end_time']), verbose=dict(type='bool', required=False, default=False), log_path=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        args = self.module.params
        self.asup = args['state'] == 'enabled'
        self.active = args['active']
        self.days = args['days']
        self.start = args['start']
        self.end = args['end']
        self.verbose = args['verbose']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.check_mode = self.module.check_mode
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        if self.start >= self.end:
            self.module.fail_json(msg='The value provided for the start time is invalid. It must be less than the end time.')
        if self.start < 0 or self.start > 23:
            self.module.fail_json(msg='The value provided for the start time is invalid. It must be between 0 and 23.')
        else:
            self.start = self.start * 60
        if self.end < 1 or self.end > 24:
            self.module.fail_json(msg='The value provided for the end time is invalid. It must be between 1 and 24.')
        else:
            self.end = min(self.end * 60, 1439)
        if not self.days:
            self.days = self.DAYS_OPTIONS

    def get_configuration(self):
        try:
            rc, result = request(self.url + 'device-asup', headers=HEADERS, **self.creds)
            if not (result['asupCapable'] and result['onDemandCapable']):
                self.module.fail_json(msg='ASUP is not supported on this device. Array Id [%s].' % self.ssid)
            return result
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve ASUP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update_configuration(self):
        config = self.get_configuration()
        update = False
        body = dict()
        if self.asup:
            body = dict(asupEnabled=True)
            if not config['asupEnabled']:
                update = True
            if (config['onDemandEnabled'] and config['remoteDiagsEnabled']) != self.active:
                update = True
                body.update(dict(onDemandEnabled=self.active, remoteDiagsEnabled=self.active))
            self.days.sort()
            config['schedule']['daysOfWeek'].sort()
            body['schedule'] = dict(daysOfWeek=self.days, dailyMinTime=self.start, dailyMaxTime=self.end, weeklyMinTime=self.start, weeklyMaxTime=self.end)
            if self.days != config['schedule']['daysOfWeek']:
                update = True
            if self.start != config['schedule']['dailyMinTime'] or self.start != config['schedule']['weeklyMinTime']:
                update = True
            elif self.end != config['schedule']['dailyMaxTime'] or self.end != config['schedule']['weeklyMaxTime']:
                update = True
        elif config['asupEnabled']:
            body = dict(asupEnabled=False)
            update = True
        self._logger.info(pformat(body))
        if update and (not self.check_mode):
            try:
                rc, result = request(self.url + 'device-asup', method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
            except Exception as err:
                self.module.fail_json(msg='We failed to set the storage-system name! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def update(self):
        update = self.update_configuration()
        cfg = self.get_configuration()
        if self.verbose:
            self.module.exit_json(msg='The ASUP settings have been updated.', changed=update, asup=cfg['asupEnabled'], active=cfg['onDemandEnabled'], cfg=cfg)
        else:
            self.module.exit_json(msg='The ASUP settings have been updated.', changed=update, asup=cfg['asupEnabled'], active=cfg['onDemandEnabled'])

    def __call__(self, *args, **kwargs):
        self.update()