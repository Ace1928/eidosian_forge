from __future__ import (absolute_import, division, print_function)
import re
import socket
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
class PlatformFactCollector(BaseFactCollector):
    name = 'platform'
    _fact_ids = set(['system', 'kernel', 'kernel_version', 'machine', 'python_version', 'architecture', 'machine_id'])

    def collect(self, module=None, collected_facts=None):
        platform_facts = {}
        platform_facts['system'] = platform.system()
        platform_facts['kernel'] = platform.release()
        platform_facts['kernel_version'] = platform.version()
        platform_facts['machine'] = platform.machine()
        platform_facts['python_version'] = platform.python_version()
        platform_facts['fqdn'] = socket.getfqdn()
        platform_facts['hostname'] = platform.node().split('.')[0]
        platform_facts['nodename'] = platform.node()
        platform_facts['domain'] = '.'.join(platform_facts['fqdn'].split('.')[1:])
        arch_bits = platform.architecture()[0]
        platform_facts['userspace_bits'] = arch_bits.replace('bit', '')
        if platform_facts['machine'] == 'x86_64':
            platform_facts['architecture'] = platform_facts['machine']
            if platform_facts['userspace_bits'] == '64':
                platform_facts['userspace_architecture'] = 'x86_64'
            elif platform_facts['userspace_bits'] == '32':
                platform_facts['userspace_architecture'] = 'i386'
        elif solaris_i86_re.search(platform_facts['machine']):
            platform_facts['architecture'] = 'i386'
            if platform_facts['userspace_bits'] == '64':
                platform_facts['userspace_architecture'] = 'x86_64'
            elif platform_facts['userspace_bits'] == '32':
                platform_facts['userspace_architecture'] = 'i386'
        else:
            platform_facts['architecture'] = platform_facts['machine']
        if platform_facts['system'] == 'AIX':
            getconf_bin = module.get_bin_path('getconf')
            if getconf_bin:
                rc, out, err = module.run_command([getconf_bin, 'MACHINE_ARCHITECTURE'])
                data = out.splitlines()
                platform_facts['architecture'] = data[0]
            else:
                bootinfo_bin = module.get_bin_path('bootinfo')
                rc, out, err = module.run_command([bootinfo_bin, '-p'])
                data = out.splitlines()
                platform_facts['architecture'] = data[0]
        elif platform_facts['system'] == 'OpenBSD':
            platform_facts['architecture'] = platform.uname()[5]
        machine_id = get_file_content('/var/lib/dbus/machine-id') or get_file_content('/etc/machine-id')
        if machine_id:
            machine_id = machine_id.splitlines()[0]
            platform_facts['machine_id'] = machine_id
        return platform_facts