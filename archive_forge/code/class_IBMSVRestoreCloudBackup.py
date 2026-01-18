from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVRestoreCloudBackup:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(target_volume_name=dict(type='str', required=True), source_volume_uid=dict(type='str'), generation=dict(type='int'), restoreuid=dict(type='bool'), deletelatergenerations=dict(type='bool'), cancel=dict(type='bool')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.target_volume_name = self.module.params.get('target_volume_name', '')
        self.source_volume_uid = self.module.params.get('source_volume_uid', '')
        self.generation = self.module.params.get('generation', '')
        self.restoreuid = self.module.params.get('restoreuid', '')
        self.deletelatergenerations = self.module.params.get('deletelatergenerations', False)
        self.cancel = self.module.params.get('cancel', False)
        self.basic_checks()
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.target_volume_name:
            self.module.fail_json(msg='Missing mandatory parameter: target_volume_name')
        if self.cancel:
            invalids = ('source_volume_uid', 'generation', 'restoreuid', 'deletelatergenerations')
            invalid_exists = ', '.join((var for var in invalids if getattr(self, var) not in {'', None}))
            if invalid_exists:
                self.module.fail_json(msg='Parameters not supported during restore cancellation: {0}'.format(invalid_exists))

    def validate(self):
        if not self.cancel:
            cmd = 'lsvolumebackupgeneration'
            cmdargs = None
            cmdopts = {}
            if self.source_volume_uid:
                cmdopts['uid'] = self.source_volume_uid
            else:
                cmdopts['volume'] = self.target_volume_name
            result = self.restapi.svc_obj_info(cmd=cmd, cmdopts=cmdopts, cmdargs=cmdargs)
        else:
            result = True
            cmd = 'lsvdisk'
            vdata = {}
            data = self.restapi.svc_obj_info(cmd=cmd, cmdopts=None, cmdargs=[self.target_volume_name])
            if isinstance(data, list):
                for d in data:
                    vdata.update(d)
            else:
                vdata = data
            if vdata and self.cancel and (vdata['restore_status'] in {'none', 'available'}):
                self.module.exit_json(msg='No restore operation is in progress for the volume ({0}).'.format(self.target_volume_name))
        return result

    def restore_volume(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'restorevolume'
        cmdargs = [self.target_volume_name]
        cmdopts = {}
        if self.cancel:
            cmdopts['cancel'] = self.cancel
            self.msg = 'Restore operation on volume ({0}) cancelled.'.format(self.target_volume_name)
        else:
            if self.source_volume_uid:
                cmdopts['fromuid'] = self.source_volume_uid
            if self.generation:
                cmdopts['generation'] = self.generation
            if self.restoreuid:
                cmdopts['restoreuid'] = self.restoreuid
            if self.deletelatergenerations:
                cmdopts['deletelatergenerations'] = self.deletelatergenerations
            self.msg = 'Restore operation on volume ({0}) started.'.format(self.target_volume_name)
        response = self.restapi._svc_token_wrap(cmd, cmdopts, cmdargs=cmdargs)
        self.log('response=%s', response)
        self.changed = True
        if response['out']:
            if b'CMMVC9103E' in response['out']:
                self.msg = 'CMMVC9103E: Volume ({0}) is not ready to perform any operation right now.'.format(self.target_volume_name)
                self.changed = False
            elif b'CMMVC9099E' in response['out']:
                self.msg = 'No restore operation is in progress for the volume ({0}).'.format(self.target_volume_name)
                self.changed = False
            else:
                self.module.fail_json(msg=response)

    def apply(self):
        if self.validate():
            self.restore_volume()
            self.log(self.msg)
        else:
            self.msg = 'No backup exist for the given source UID/volume.'
            self.log(self.msg)
            self.module.fail_json(msg=self.msg)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
            self.log(self.msg)
        self.module.exit_json(changed=self.changed, msg=self.msg)