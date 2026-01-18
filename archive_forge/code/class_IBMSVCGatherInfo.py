from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCGatherInfo(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(objectname=dict(type='str'), gather_subset=dict(type='list', elements='str', required=False, default=['all'], choices=['vol', 'pool', 'node', 'iog', 'host', 'hostvdiskmap', 'vdiskhostmap', 'hc', 'fc', 'fcport', 'targetportfc', 'iscsiport', 'fcmap', 'rcrelationship', 'fcconsistgrp', 'rcconsistgrp', 'vdiskcopy', 'array', 'system', 'cloudaccount', 'cloudaccountusage', 'cloudimportcandidate', 'ldapserver', 'drive', 'user', 'usergroup', 'ownershipgroup', 'partnership', 'replicationpolicy', 'cloudbackup', 'cloudbackupgeneration', 'snapshotpolicy', 'snapshotpolicyschedule', 'volumegroup', 'volumegroupsnapshotpolicy', 'volumesnapshot', 'dnsserver', 'systemcertificate', 'truststore', 'sra', 'syslogserver', 'emailserver', 'emailuser', 'provisioningpolicy', 'volumegroupsnapshot', 'callhome', 'ip', 'portset', 'safeguardedpolicy', 'mdisk', 'safeguardedpolicyschedule', 'eventlog', 'enclosurestats', 'enclosurestatshistory', 'all'])))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        self.log = get_logger(self.__class__.__name__, log_path)
        self.objectname = self.module.params['objectname']
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def validate(self, subset):
        if not self.objectname:
            self.module.fail_json(msg='Following paramter is mandatory to execute {0}: objectname'.format(subset))

    @property
    def cloudbackupgeneration(self):
        return self.restapi.svc_obj_info(cmd='lsvolumebackupgeneration', cmdopts={'volume': self.objectname}, cmdargs=None)

    @property
    def enclosurestatshistory(self):
        return self.restapi.svc_obj_info(cmd='lsenclosurestats', cmdopts={'history': 'power_w:temp_c:temp_f'}, cmdargs=[self.objectname])

    def get_list(self, subset, op_key, cmd, validate):
        try:
            if validate:
                self.validate(subset)
            output = {}
            exceptions = {'cloudbackupgeneration', 'enclosurestatshistory'}
            if subset in exceptions:
                output[op_key] = getattr(self, subset)
            else:
                cmdargs = [self.objectname] if self.objectname else None
                output[op_key] = self.restapi.svc_obj_info(cmd=cmd, cmdopts=None, cmdargs=cmdargs)
            self.log.info('Successfully listed %d %s info from cluster %s', len(subset), subset, self.module.params['clustername'])
            return output
        except Exception as e:
            msg = 'Get %s info from cluster %s failed with error %s ' % (subset, self.module.params['clustername'], str(e))
            self.log.error(msg)
            self.module.fail_json(msg=msg)

    def apply(self):
        subset = self.module.params['gather_subset']
        if self.objectname and len(subset) != 1:
            msg = 'objectname(%s) is specified while gather_subset(%s) is not one of %s' % (self.objectname, self.subset, all)
            self.module.fail_json(msg=msg)
        if len(subset) == 0 or 'all' in subset:
            self.log.info('The default value for gather_subset is all')
        result = {'Volume': [], 'Pool': [], 'Node': [], 'IOGroup': [], 'Host': [], 'HostVdiskMap': [], 'VdiskHostMap': [], 'HostCluster': [], 'FCConnectivitie': [], 'FCConsistgrp': [], 'RCConsistgrp': [], 'VdiskCopy': [], 'FCPort': [], 'TargetPortFC': [], 'iSCSIPort': [], 'FCMap': [], 'RemoteCopy': [], 'Array': [], 'System': [], 'CloudAccount': [], 'CloudAccountUsage': [], 'CloudImportCandidate': [], 'LdapServer': [], 'Drive': [], 'User': [], 'Partnership': [], 'ReplicationPolicy': [], 'SnapshotPolicy': [], 'VolumeGroup': [], 'SnapshotSchedule': [], 'VolumeGroupSnapshotPolicy': [], 'VolumeSnapshot': [], 'DnsServer': [], 'SystemCert': [], 'TrustStore': [], 'Sra': [], 'SysLogServer': [], 'UserGrp': [], 'EmailServer': [], 'EmailUser': [], 'CloudBackup': [], 'CloudBackupGeneration': [], 'ProvisioningPolicy': [], 'VolumeGroupSnapshot': [], 'CallHome': [], 'IP': [], 'Ownershipgroup': [], 'Portset': [], 'SafeguardedPolicy': [], 'Mdisk': [], 'SafeguardedSchedule': [], 'EventLog': []}
        cmd_mappings = {'vol': ('Volume', 'lsvdisk', False), 'pool': ('Pool', 'lsmdiskgrp', False), 'node': ('Node', 'lsnode', False), 'iog': ('IOGroup', 'lsiogrp', False), 'host': ('Host', 'lshost', False), 'hostvdiskmap': ('HostVdiskMap', 'lshostvdiskmap', False), 'vdiskhostmap': ('VdiskHostMap', 'lsvdiskhostmap', False), 'hc': ('HostCluster', 'lshostcluster', False), 'fc': ('FCConnectivitie', 'lsfabric', False), 'fcport': ('FCPort', 'lsportfc', False), 'iscsiport': ('iSCSIPort', 'lsportip', False), 'fcmap': ('FCMap', 'lsfcmap', False), 'rcrelationship': ('RemoteCopy', 'lsrcrelationship', False), 'fcconsistgrp': ('FCConsistgrp', 'lsfcconsistgrp', False), 'rcconsistgrp': ('RCConsistgrp', 'lsrcconsistgrp', False), 'vdiskcopy': ('VdiskCopy', 'lsvdiskcopy', False), 'targetportfc': ('TargetPortFC', 'lstargetportfc', False), 'array': ('Array', 'lsarray', False), 'system': ('System', 'lssystem', False), 'cloudaccount': ('CloudAccount', 'lscloudaccount', False), 'cloudaccountusage': ('CloudAccountUsage', 'lscloudaccountusage', False), 'cloudimportcandidate': ('CloudImportCandidate', 'lscloudaccountimportcandidate', False), 'ldapserver': ('LdapServer', 'lsldapserver', False), 'drive': ('Drive', 'lsdrive', False), 'user': ('User', 'lsuser', False), 'usergroup': ('UserGrp', 'lsusergrp', False), 'ownershipgroup': ('Ownershipgroup', 'lsownershipgroup', False), 'partnership': ('Partnership', 'lspartnership', False), 'replicationpolicy': ('ReplicationPolicy', 'lsreplicationpolicy', False), 'cloudbackup': ('CloudBackup', 'lsvolumebackup', False), 'cloudbackupgeneration': ('CloudBackupGeneration', 'lsvolumebackupgeneration', True), 'snapshotpolicy': ('SnapshotPolicy', 'lssnapshotpolicy', False), 'snapshotpolicyschedule': ('SnapshotSchedule', 'lssnapshotschedule', False), 'volumegroup': ('VolumeGroup', 'lsvolumegroup', False), 'volumegroupsnapshotpolicy': ('VolumeGroupSnapshotPolicy', 'lsvolumegroupsnapshotpolicy', False), 'volumesnapshot': ('VolumeSnapshot', 'lsvolumesnapshot', False), 'dnsserver': ('DnsServer', 'lsdnsserver', False), 'systemcertificate': ('SystemCert', 'lssystemcert', False), 'truststore': ('TrustStore', 'lstruststore', False), 'sra': ('Sra', 'lssra', False), 'syslogserver': ('SysLogServer', 'lssyslogserver', False), 'emailserver': ('EmailServer', 'lsemailserver', False), 'emailuser': ('EmailUser', 'lsemailuser', False), 'provisioningpolicy': ('ProvisioningPolicy', 'lsprovisioningpolicy', False), 'volumegroupsnapshot': ('VolumeGroupSnapshot', 'lsvolumegroupsnapshot', False), 'callhome': ('CallHome', 'lscloudcallhome', False), 'ip': ('IP', 'lsip', False), 'portset': ('Portset', 'lsportset', False), 'safeguardedpolicy': ('SafeguardedPolicy', 'lssafeguardedpolicy', False), 'mdisk': ('Mdisk', 'lsmdisk', False), 'safeguardedpolicyschedule': ('SafeguardedSchedule', 'lssafeguardedschedule', False), 'eventlog': ('EventLog', 'lseventlog', False), 'enclosurestats': ('EnclosureStats', 'lsenclosurestats', False), 'enclosurestatshistory': ('EnclosureStatsHistory', 'lsenclosurestats -history power_w:temp_c:temp_f', True)}
        if subset == ['all']:
            current_set = cmd_mappings.keys()
        else:
            current_set = subset
        for key in current_set:
            value_tuple = cmd_mappings[key]
            if subset == ['all'] and value_tuple[2]:
                continue
            op = self.get_list(key, *value_tuple)
            result.update(op)
        self.module.exit_json(**result)