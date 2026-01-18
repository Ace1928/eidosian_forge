from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVFCPortsetmember:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=True, choices=['present', 'absent']), name=dict(type='str', required=True), fcportid=dict(type='str', required=True)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.fcportid = self.module.params['fcportid']
        self.basic_checks()
        self.fcportsetmember_details = None
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if not self.fcportid:
            self.module.fail_json(msg='Missing mandatory parameter: fcportid ')

    def is_fcportsetmember_exists(self):
        merged_result = {}
        cmd = 'lsfcportsetmember'
        cmdopts = {'filtervalue': 'portset_name={0}:fc_io_port_id={1}'.format(self.name, self.fcportid)}
        data = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        self.fcportsetmember_details = merged_result
        return merged_result

    def add_fcportsetmember(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'addfcportsetmember'
        cmdopts = {'portset': self.name, 'fcioportid': self.fcportid}
        self.changed = True
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('FCPortsetmember (%s) mapping is created with fcportid (%s) successfully.', self.name, self.fcportid)

    def remove_fcportsetmember(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmfcportsetmember'
        cmdopts = {'portset': self.name, 'fcioportid': self.fcportid}
        self.changed = True
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('FCPortsetmember (%s) mapping is removed from fcportid (%s) successfully.', self.name, self.fcportid)

    def apply(self):
        fcportsetmember_data = self.is_fcportsetmember_exists()
        if fcportsetmember_data:
            if self.state == 'present':
                self.msg = 'FCPortsetmember ({0}) mapping with fcportid ({1}) is already exist.'.format(self.name, self.fcportid)
            else:
                self.remove_fcportsetmember()
                self.msg = 'FCPortsetmember ({0}) mapping is removed from fcportid ({1}) successfully.'.format(self.name, self.fcportid)
        elif self.state == 'absent':
            self.msg = 'FCPortsetmember ({0}) mapping does not exist with fcportid ({1}). No modifications done.'.format(self.name, self.fcportid)
        else:
            self.add_fcportsetmember()
            self.msg = 'FCPortsetmember ({0}) mapping is created with fcportid ({1}) successfully.'.format(self.name, self.fcportid)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)