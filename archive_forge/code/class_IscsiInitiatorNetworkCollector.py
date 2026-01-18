from __future__ import (absolute_import, division, print_function)
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.network.base import NetworkCollector
class IscsiInitiatorNetworkCollector(NetworkCollector):
    name = 'iscsi'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        """
        Example of contents of /etc/iscsi/initiatorname.iscsi:

        ## DO NOT EDIT OR REMOVE THIS FILE!
        ## If you remove this file, the iSCSI daemon will not start.
        ## If you change the InitiatorName, existing access control lists
        ## may reject this initiator.  The InitiatorName must be unique
        ## for each iSCSI initiator.  Do NOT duplicate iSCSI InitiatorNames.
        InitiatorName=iqn.1993-08.org.debian:01:44a42c8ddb8b

        Example of output from the AIX lsattr command:

        # lsattr -E -l iscsi0
        disc_filename  /etc/iscsi/targets            Configuration file                            False
        disc_policy    file                          Discovery Policy                              True
        initiator_name iqn.localhost.hostid.7f000002 iSCSI Initiator Name                          True
        isns_srvnames  auto                          iSNS Servers IP Addresses                     True
        isns_srvports                                iSNS Servers Port Numbers                     True
        max_targets    16                            Maximum Targets Allowed                       True
        num_cmd_elems  200                           Maximum number of commands to queue to driver True

        Example of output from the HP-UX iscsiutil command:

        #iscsiutil -l
        Initiator Name             : iqn.1986-03.com.hp:mcel_VMhost3.1f355cf6-e2db-11e0-a999-b44c0aef5537
        Initiator Alias            :

        Authentication Method      : None
        CHAP Method                : CHAP_UNI
        Initiator CHAP Name        :
        CHAP Secret                :
        NAS Hostname               :
        NAS Secret                 :
        Radius Server Hostname     :
        Header Digest              : None, CRC32C (default)
        Data Digest                : None, CRC32C (default)
        SLP Scope list for iSLPD   :
        """
        iscsi_facts = {}
        iscsi_facts['iscsi_iqn'] = ''
        if sys.platform.startswith('linux') or sys.platform.startswith('sunos'):
            for line in get_file_content('/etc/iscsi/initiatorname.iscsi', '').splitlines():
                if line.startswith('#') or line.startswith(';') or line.strip() == '':
                    continue
                if line.startswith('InitiatorName='):
                    iscsi_facts['iscsi_iqn'] = line.split('=', 1)[1]
                    break
        elif sys.platform.startswith('aix'):
            try:
                cmd = get_bin_path('lsattr')
            except ValueError:
                return iscsi_facts
            cmd += ' -E -l iscsi0'
            rc, out, err = module.run_command(cmd)
            if rc == 0 and out:
                line = self.findstr(out, 'initiator_name')
                iscsi_facts['iscsi_iqn'] = line.split()[1].rstrip()
        elif sys.platform.startswith('hp-ux'):
            try:
                cmd = get_bin_path('iscsiutil', opt_dirs=['/opt/iscsi/bin'])
            except ValueError:
                return iscsi_facts
            cmd += ' -l'
            rc, out, err = module.run_command(cmd)
            if out:
                line = self.findstr(out, 'Initiator Name')
                iscsi_facts['iscsi_iqn'] = line.split(':', 1)[1].rstrip()
        return iscsi_facts

    def findstr(self, text, match):
        for line in text.splitlines():
            if match in line:
                found = line
        return found