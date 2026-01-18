from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
class IBMSVCmdisk(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present']), level=dict(type='str', choices=['raid0', 'raid1', 'raid5', 'raid6', 'raid10']), drive=dict(type='str', default=None), encrypt=dict(type='str', default='no', choices=['yes', 'no']), mdiskgrp=dict(type='str', required=True)))
        mutually_exclusive = []
        self.module = AnsibleModule(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.level = self.module.params.get('level', None)
        self.drive = self.module.params.get('drive', None)
        self.encrypt = self.module.params.get('encrypt', None)
        self.mdiskgrp = self.module.params.get('mdiskgrp', None)
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def mdisk_exists(self):
        return self.restapi.svc_obj_info(cmd='lsmdisk', cmdopts=None, cmdargs=[self.name])

    def mdisk_create(self):
        if not self.level:
            self.module.fail_json(msg='You must pass in level to the module.')
        if not self.drive:
            self.module.fail_json(msg='You must pass in drive to the module.')
        if not self.mdiskgrp:
            self.module.fail_json(msg='You must pass in mdiskgrp to the module.')
        if self.module.check_mode:
            self.changed = True
            return
        self.log("creating mdisk '%s'", self.name)
        cmd = 'mkarray'
        cmdopts = {}
        if self.level:
            cmdopts['level'] = self.level
        if self.drive:
            cmdopts['drive'] = self.drive
        if self.encrypt:
            cmdopts['encrypt'] = self.encrypt
        cmdopts['name'] = self.name
        cmdargs = [self.mdiskgrp]
        self.log('creating mdisk command=%s opts=%s args=%s', cmd, cmdopts, cmdargs)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.log('create mdisk result %s', result)
        if 'message' in result:
            self.changed = True
            self.log('create mdisk result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create mdisk [%s]' % self.name)

    def mdisk_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting mdisk '%s'", self.name)
        cmd = 'rmmdisk'
        cmdopts = {}
        cmdopts['mdisk'] = self.name
        cmdargs = [self.mdiskgrp]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def mdisk_update(self, modify):
        self.log("updating mdisk '%s'", self.name)
        self.changed = True

    def mdisk_probe(self, data):
        props = []
        if self.encrypt:
            if self.encrypt != data['encrypt']:
                props += ['encrypt']
        if props is []:
            props = None
        self.log("mdisk_probe props='%s'", data)
        return props

    def apply(self):
        changed = False
        msg = None
        modify = []
        mdisk_data = self.mdisk_exists()
        if mdisk_data:
            if self.state == 'absent':
                self.log("CHANGED: mdisk exists, but requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                modify = self.mdisk_probe(mdisk_data)
                if modify:
                    changed = True
        elif self.state == 'present':
            self.log("CHANGED: mdisk does not exist, but requested state is 'present'")
            changed = True
        if changed:
            if self.state == 'present':
                if not mdisk_data:
                    self.mdisk_create()
                    msg = 'Mdisk [%s] has been created.' % self.name
                else:
                    self.mdisk_update(modify)
                    msg = 'Mdisk [%s] has been modified.' % self.name
            elif self.state == 'absent':
                self.mdisk_delete()
                msg = 'Volume [%s] has been deleted.' % self.name
            if self.module.check_mode:
                msg = 'skipping changes due to check mode'
        else:
            self.log('exiting with no changes')
            if self.state == 'absent':
                msg = 'Mdisk [%s] did not exist.' % self.name
            else:
                msg = 'Mdisk [%s] already exists.' % self.name
        self.module.exit_json(msg=msg, changed=changed)