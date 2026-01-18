from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
class IBMSVCOwnershipgroup:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['present', 'absent']), keepobjects=dict(type='bool')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.keepobjects = self.module.params.get('keepobjects')
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        log_path = self.module.params['log_path']
        logger = get_logger(self.__class__.__name__, log_path)
        self.log = logger.info
        self.changed = False
        self.msg = None
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def check_existing_owgroups(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsownershipgroup', cmdopts=None, cmdargs=[self.name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        return merged_result

    def create_ownershipgroup(self):
        if self.module.check_mode:
            self.changed = True
            return
        if self.keepobjects:
            self.module.fail_json(msg='Keepobjects should only be passed while deleting ownershipgroup')
        cmd = 'mkownershipgroup'
        cmdopts = None
        cmdargs = ['-name', self.name]
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
        self.log('Create ownership group result: %s', result)

    def delete_ownershipgroup(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmownershipgroup'
        cmdopts = None
        cmdargs = [self.name]
        if self.keepobjects:
            cmdargs.insert(0, '-keepobjects')
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
        self.log('Delete ownership group result: %s', result)

    def apply(self):
        if self.check_existing_owgroups():
            if self.state == 'present':
                self.msg = 'Ownership group (%s) already exist.' % self.name
            else:
                self.delete_ownershipgroup()
                self.msg = 'Ownership group (%s) deleted.' % self.name
        elif self.state == 'absent':
            self.msg = 'Ownership group (%s) does not exist.' % self.name
        else:
            self.create_ownershipgroup()
            self.msg = 'Ownership group (%s) created.' % self.name
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)