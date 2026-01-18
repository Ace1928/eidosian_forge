from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCvdiskhostmap(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(volname=dict(type='str', required=True), host=dict(type='str', required=False), state=dict(type='str', required=True, choices=['absent', 'present']), scsi=dict(type='int', required=False), hostcluster=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.volname = self.module.params['volname']
        self.state = self.module.params['state']
        self.host = self.module.params['host']
        self.hostcluster = self.module.params['hostcluster']
        self.scsi = self.module.params['scsi']
        if not self.volname:
            self.module.fail_json(msg='Missing mandatory parameter: volname')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def get_existing_vdiskhostmap(self):
        merged_result = []
        data = self.restapi.svc_obj_info(cmd='lsvdiskhostmap', cmdopts=None, cmdargs=[self.volname])
        if isinstance(data, list):
            for d in data:
                merged_result.append(d)
        elif data:
            merged_result = [data]
        return merged_result

    def vdiskhostmap_probe(self, mdata):
        props = []
        self.log("vdiskhostmap_probe props='%s'", mdata)
        mapping_exist = False
        for data in mdata:
            if self.host:
                if self.host == data['host_name'] and self.volname == data['name']:
                    if self.scsi and self.scsi != int(data['SCSI_id']):
                        self.module.fail_json(msg='Update not supported for parameter: scsi')
                    mapping_exist = True
            elif self.hostcluster:
                if self.hostcluster == data['host_cluster_name'] and self.volname == data['name']:
                    if self.scsi and self.scsi != int(data['SCSI_id']):
                        self.module.fail_json(msg='Update not supported for parameter: scsi')
                    mapping_exist = True
        if not mapping_exist:
            props += ['map']
        if props is []:
            props = None
        self.log("vdiskhostmap_probe props='%s'", props)
        return props

    def vdiskhostmap_create(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("creating vdiskhostmap '%s' '%s'", self.volname, self.host)
        cmd = 'mkvdiskhostmap'
        cmdopts = {'force': True}
        cmdopts['host'] = self.host
        cmdopts['scsi'] = self.scsi
        cmdargs = [self.volname]
        self.log('creating vdiskhostmap command %s opts %s args %s', cmd, cmdopts, cmdargs)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.log('create vdiskhostmap result %s', result)
        if 'message' in result:
            self.changed = True
            self.log('create vdiskhostmap result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create vdiskhostmap.')

    def vdiskhostmap_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting vdiskhostmap '%s'", self.volname)
        cmd = 'rmvdiskhostmap'
        cmdopts = {}
        cmdopts['host'] = self.host
        cmdargs = [self.volname]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def vdiskhostclustermap_create(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("creating mkvolumehostclustermap '%s' '%s'", self.volname, self.hostcluster)
        cmd = 'mkvolumehostclustermap'
        cmdopts = {'force': True}
        cmdopts['hostcluster'] = self.hostcluster
        cmdopts['scsi'] = self.scsi
        cmdargs = [self.volname]
        self.log('creating vdiskhostmap command %s opts %s args %s', cmd, cmdopts, cmdargs)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.log('create vdiskhostmap result %s', result)
        if 'message' in result:
            self.changed = True
            self.log('create vdiskhostmap result message %s', result['message'])
        else:
            self.module.fail_json(msg='Failed to create vdiskhostmap.')

    def vdiskhostclustermap_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting vdiskhostclustermap '%s'", self.volname)
        cmd = 'rmvolumehostclustermap'
        cmdopts = {}
        cmdopts['hostcluster'] = self.hostcluster
        cmdargs = [self.volname]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def apply(self):
        changed = False
        msg = None
        if not self.volname:
            self.module.fail_json(msg='You must pass in volname to the module.')
        if self.host and self.hostcluster:
            self.module.fail_json(msg='Either use host or hostcluster')
        elif not self.host and (not self.hostcluster):
            self.module.fail_json(msg='Missing parameter: host or hostcluster')
        vdiskmap_data = self.get_existing_vdiskhostmap()
        self.log("volume mapping data is : '%s'", vdiskmap_data)
        if vdiskmap_data:
            if self.state == 'absent':
                self.log("vdiskmap exists, and requested state is 'absent'")
                changed = True
            elif self.state == 'present':
                probe_data = self.vdiskhostmap_probe(vdiskmap_data)
                if probe_data:
                    self.log("vdiskmap does not exist, but requested state is 'present'")
                    changed = True
        elif self.state == 'present':
            self.log("vdiskmap does not exist, but requested state is 'present'")
            changed = True
        if changed:
            if self.state == 'present':
                if self.host:
                    self.vdiskhostmap_create()
                    msg = 'Vdiskhostmap %s %s has been created.' % (self.volname, self.host)
                elif self.hostcluster:
                    self.vdiskhostclustermap_create()
                    msg = 'Vdiskhostclustermap %s %s has been created.' % (self.volname, self.hostcluster)
            elif self.state == 'absent':
                if self.host:
                    self.vdiskhostmap_delete()
                    msg = 'vdiskhostmap [%s] has been deleted.' % self.volname
                elif self.hostcluster:
                    self.vdiskhostclustermap_delete()
                    msg = 'vdiskhostclustermap [%s] has been deleted.' % self.volname
            if self.module.check_mode:
                msg = 'skipping changes due to check mode'
        else:
            self.log('exiting with no changes')
            if self.state == 'absent':
                msg = 'Volume mapping [%s] did not exist.' % self.volname
            else:
                msg = 'Volume mapping [%s] already exists.' % self.volname
        self.module.exit_json(msg=msg, changed=changed)