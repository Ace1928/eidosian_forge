from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _get_local_tz(module, timezone='UTC'):
    """
    We will attempt to get the local timezone of the server running the module and use that.
    If we can't get the timezone then we will set the default to be UTC

    Linnux has been tested and other opersting systems should be OK.
    Failures cause assumption of UTC

    Windows is not supported and will assume UTC
    """
    if platform.system() == 'Linux':
        timedatectl = get_bin_path('timedatectl')
        if timedatectl is not None:
            rcode, stdout, stderr = module.run_command(timedatectl)
            if rcode == 0 and stdout:
                line = _findstr(stdout, 'Time zone')
                full_tz = line.split(':', 1)[1].rstrip()
                timezone = full_tz.split()[0]
                return timezone
            else:
                module.warn('Incorrect timedatectl output. Timezone will be set to UTC')
        elif os.path.exists('/etc/timezone'):
            timezone = get_file_content('/etc/timezone')
        else:
            module.warn('Could not find /etc/timezone. Assuming UTC')
    elif platform.system() == 'SunOS':
        if os.path.exists('/etc/default/init'):
            for line in get_file_content('/etc/default/init', '').splitlines():
                if line.startswith('TZ='):
                    timezone = line.split('=', 1)[1]
                    return timezone
        else:
            module.warn('Could not find /etc/default/init. Assuming UTC')
    elif re.match('^Darwin', platform.platform()):
        systemsetup = get_bin_path('systemsetup')
        if systemsetup is not None:
            rcode, stdout, stderr = module.execute(systemsetup, '-gettimezone')
            if rcode == 0 and stdout:
                timezone = stdout.split(':', 1)[1].lstrip()
            else:
                module.warn('Could not run systemsetup. Assuming UTC')
        else:
            module.warn('Could not find systemsetup. Assuming UTC')
    elif re.match('^(Free|Net|Open)BSD', platform.platform()):
        if os.path.exists('/etc/timezone'):
            timezone = get_file_content('/etc/timezone')
        else:
            module.warn('Could not find /etc/timezone. Assuming UTC')
    elif platform.system() == 'AIX':
        aix_oslevel = int(platform.version() + platform.release())
        if aix_oslevel >= 61:
            if os.path.exists('/etc/environment'):
                for line in get_file_content('/etc/environment', '').splitlines():
                    if line.startswith('TZ='):
                        timezone = line.split('=', 1)[1]
                        return timezone
            else:
                module.warn('Could not find /etc/environment. Assuming UTC')
        else:
            module.warn('Cannot determine timezone when AIX os level < 61. Assuming UTC')
    else:
        module.warn('Could not find /etc/timezone. Assuming UTC')
    return timezone