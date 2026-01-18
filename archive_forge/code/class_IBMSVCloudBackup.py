from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVCloudBackup:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', choices=['present', 'absent'], required=True), volume_name=dict(type='str'), volumegroup_name=dict(type='str'), generation=dict(type='int'), volume_UID=dict(type='str'), full=dict(type='bool'), all=dict(type='bool')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.state = self.module.params['state']
        self.volume_name = self.module.params.get('volume_name')
        self.volumegroup_name = self.module.params.get('volumegroup_name')
        self.full = self.module.params.get('full')
        self.volume_UID = self.module.params.get('volume_UID', '')
        self.generation = self.module.params.get('generation', '')
        self.all = self.module.params.get('all')
        self.basic_checks()
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if self.state == 'present':
            if self.volume_UID:
                self.module.fail_json(msg='Parameter not supported during creation: volume_UID')
            if self.volume_name and self.volumegroup_name:
                self.module.fail_json(msg='Mutually exclusive parameters: volume_name, volumegroup_name')
            if not self.volumegroup_name and (not self.volume_name):
                self.module.fail_json(msg='One of these parameter required to create backup: volume_name, volumegroup_name')
            invalids = ('generation', 'all')
            invalid_exists = ', '.join((var for var in invalids if getattr(self, var) not in {'', None}))
            if invalid_exists:
                self.module.fail_json(msg='Following parameters not supported during creation: {0}'.format(invalid_exists))
        else:
            if self.volume_name and self.volume_UID:
                self.module.fail_json(msg='Mutually exclusive parameters: volume_name, volume_UID')
            if not self.volume_name and (not self.volume_UID):
                self.module.fail_json(msg='One of these parameter required to delete backup: volume_name, volume_UID')
            if self.generation and self.all:
                self.module.fail_json(msg='Mutually exclusive parameters: generation, all')
            if self.generation in {'', None} and self.all in {'', None}:
                self.module.fail_json(msg='One of the following parameter is required: generation, all')
            if self.volumegroup_name:
                self.module.fail_json(msg='Parameter not supported during deletion: volumegroup_name')
            if self.full not in {'', None}:
                self.module.fail_json(msg='Parameter not supported during deletion: full')

    def check_source(self):
        result = {}
        if self.volumegroup_name:
            cmd = 'lsvolumegroup'
            cmdargs = [self.volumegroup_name]
            cmdopts = None
        elif self.volume_name and self.state == 'present':
            cmd = 'lsvdisk'
            cmdargs = [self.volume_name]
            cmdopts = None
        else:
            cmd = 'lsvolumebackupgeneration'
            cmdargs = None
            cmdopts = {}
            if self.volume_UID:
                self.var = self.volume_UID
                cmdopts['uid'] = self.volume_UID
            else:
                self.var = self.volume_name
                cmdopts['volume'] = self.volume_name
        data = self.restapi.svc_obj_info(cmd=cmd, cmdopts=cmdopts, cmdargs=cmdargs)
        if isinstance(data, list):
            for d in data:
                result.update(d)
        else:
            result = data
        if self.state == 'present':
            return not result
        else:
            return result

    def create_cloud_backup(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmdopts = {}
        if self.volume_name:
            cmd = 'backupvolume'
            cmdargs = [self.volume_name]
            self.msg = 'Cloud backup ({0}) created'.format(self.volume_name)
        else:
            cmd = 'backupvolumegroup'
            cmdargs = [self.volumegroup_name]
            self.msg = 'Cloud backup ({0}) created'.format(self.volumegroup_name)
        if self.full:
            cmdopts['full'] = True
        response = self.restapi._svc_token_wrap(cmd, cmdopts, cmdargs=cmdargs)
        self.log('create_cloud_backup response=%s', response)
        self.changed = True
        if response['out']:
            if b'CMMVC9083E' in response['out']:
                self.msg = 'CMMVC9083E: Volume is not ready to perform any operation right now.'
                self.changed = False
            elif b'CMMVC8753E' in response['out']:
                self.msg = 'Backup already in progress.'
                self.changed = False
            else:
                self.msg = response
                self.module.fail_json(msg=self.msg)
        self.log(self.msg)

    def delete_cloud_backup(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmvolumebackupgeneration'
        cmdopts = {}
        if self.volume_name:
            cmdopts['volume'] = self.volume_name
            var = self.volume_name
            self.msg = 'Cloud backup ({0}) deleted'.format(self.volume_name)
        else:
            cmdopts['uid'] = self.volume_UID
            var = self.volume_UID
            self.msg = 'Cloud backup ({0}) deleted'.format(self.volume_UID)
        if self.generation:
            cmdopts['generation'] = self.generation
        if self.all not in {'', None}:
            cmdopts['all'] = self.all
        response = self.restapi._svc_token_wrap(cmd, cmdopts=cmdopts, cmdargs=None)
        self.log('response=%s', response)
        self.changed = True
        if response['out']:
            if b'CMMVC9104E' in response['out']:
                self.changed = False
                self.msg = 'CMMVC9104E: Volume ({0}) is not ready to perform any operation right now.'.format(var)
            elif b'CMMVC9090E' in response['out']:
                self.changed = False
                self.msg = 'Cloud backup generation already deleted.'
            else:
                self.module.fail_json(msg=response)
        self.log(self.msg)

    def apply(self):
        if self.check_source():
            if self.state == 'present':
                self.module.fail_json(msg='Volume (or) Volumegroup does not exist.')
            else:
                self.delete_cloud_backup()
        elif self.state == 'absent':
            self.msg = 'Backup ({0}) does not exist for the given name/UID.'.format(self.var)
            self.log(self.msg)
        else:
            self.create_cloud_backup()
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
            self.log(self.msg)
        self.module.exit_json(changed=self.changed, msg=self.msg)