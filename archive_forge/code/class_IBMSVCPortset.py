from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVCPortset:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=True, choices=['present', 'absent']), name=dict(type='str', required=True), portset_type=dict(type='str', choices=['host', 'replication']), ownershipgroup=dict(type='str'), noownershipgroup=dict(type='bool'), porttype=dict(type='str', choices=['fc', 'ethernet']), old_name=dict(type='str')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.portset_type = self.module.params.get('portset_type', '')
        self.ownershipgroup = self.module.params.get('ownershipgroup', '')
        self.noownershipgroup = self.module.params.get('noownershipgroup', '')
        self.porttype = self.module.params.get('porttype', '')
        self.old_name = self.module.params.get('old_name', '')
        self.basic_checks()
        self.portset_details = None
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if self.state == 'present':
            if not self.name:
                self.module.fail_json(msg='Missing mandatory parameter: name')
            if self.ownershipgroup and self.noownershipgroup:
                self.module.fail_json(msg='Mutually exclusive parameter: ownershipgroup, noownershipgroup')
        else:
            if not self.name:
                self.module.fail_json(msg='Missing mandatory parameter: name')
            fields = [f for f in ['ownershipgroup', 'noownershipgroup', 'porttype', 'portset_type', 'old_name'] if getattr(self, f)]
            if any(fields):
                self.module.fail_json(msg='Parameters {0} not supported while deleting a porset'.format(', '.join(fields)))

    def parameter_handling_while_renaming(self):
        parameters = {'ownershipgroup': self.ownershipgroup, 'noownershipgroup': self.noownershipgroup, 'porttype': self.porttype, 'portset_type': self.portset_type}
        parameters_exists = [parameter for parameter, value in parameters.items() if value]
        if parameters_exists:
            self.module.fail_json(msg='Parameters {0} not supported while renaming a portset.'.format(', '.join(parameters_exists)))

    def is_portset_exists(self, portset_name):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsportset', cmdopts=None, cmdargs=[portset_name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        self.portset_details = merged_result
        return merged_result

    def create_portset(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mkportset'
        cmdopts = {'name': self.name, 'type': self.portset_type if self.portset_type else 'host', 'porttype': self.porttype if self.porttype else 'ethernet'}
        if self.ownershipgroup:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('Portset (%s) created', self.name)
        self.changed = True

    def portset_probe(self):
        updates = []
        if self.portset_type and self.portset_type != self.portset_details['type']:
            self.module.fail_json(msg="portset_type can't be updated for portset")
        if self.porttype and self.porttype != self.portset_details['port_type']:
            self.module.fail_json(msg="porttype can't be updated for portset")
        if self.ownershipgroup and self.ownershipgroup != self.portset_details['owner_name']:
            updates.append('ownershipgroup')
        if self.noownershipgroup:
            updates.append('noownershipgroup')
        self.log('Modifications to be done: %s', updates)
        return updates

    def update_portset(self, updates):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'chportset'
        cmdopts = dict(((k, getattr(self, k)) for k in updates))
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts=cmdopts, cmdargs=cmdargs)
        self.log('Portset (%s) updated', self.name)
        self.changed = True

    def delete_portset(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmportset'
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts=None, cmdargs=cmdargs)
        self.log('Portset (%s) deleted', self.name)
        self.changed = True

    def portset_rename(self, portset_data):
        msg = ''
        self.parameter_handling_while_renaming()
        old_portset_data = self.is_portset_exists(self.old_name)
        if not old_portset_data and (not portset_data):
            self.module.fail_json(msg="Portset with old name {0} doesn't exist.".format(self.old_name))
        elif old_portset_data and portset_data:
            self.module.fail_json(msg='Portset [{0}] already exists.'.format(self.name))
        elif not old_portset_data and portset_data:
            msg = 'Portset with name [{0}] already exists.'.format(self.name)
        elif old_portset_data and (not portset_data):
            if self.module.check_mode:
                self.changed = True
                return
            self.restapi.svc_run_command('chportset', {'name': self.name}, [self.old_name])
            self.changed = True
            msg = 'Portset [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
        return msg

    def apply(self):
        portset_data = self.is_portset_exists(self.name)
        if self.state == 'present' and self.old_name:
            self.msg = self.portset_rename(portset_data)
        elif self.state == 'absent' and self.old_name:
            self.module.fail_json(msg="Rename functionality is not supported when 'state' is absent.")
        elif portset_data:
            if self.state == 'present':
                modifications = self.portset_probe()
                if any(modifications):
                    self.update_portset(modifications)
                    self.msg = 'Portset ({0}) updated.'.format(self.name)
                else:
                    self.msg = 'Portset ({0}) already exists. No modifications done.'.format(self.name)
            else:
                self.delete_portset()
                self.msg = 'Portset ({0}) deleted successfully.'.format(self.name)
        elif self.state == 'absent':
            self.msg = 'Portset ({0}) does not exist. No modifications done.'.format(self.name)
        else:
            self.create_portset()
            self.msg = 'Portset ({0}) created successfully.'.format(self.name)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)