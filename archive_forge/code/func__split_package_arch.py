from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _split_package_arch(self, packagename):
    redhat_rpm_arches = ['aarch64', 'alphaev56', 'alphaev5', 'alphaev67', 'alphaev6', 'alpha', 'alphapca56', 'amd64', 'armv3l', 'armv4b', 'armv4l', 'armv5tejl', 'armv5tel', 'armv5tl', 'armv6hl', 'armv6l', 'armv7hl', 'armv7hnl', 'armv7l', 'athlon', 'geode', 'i386', 'i486', 'i586', 'i686', 'ia32e', 'ia64', 'm68k', 'mips64el', 'mips64', 'mips64r6el', 'mips64r6', 'mipsel', 'mips', 'mipsr6el', 'mipsr6', 'noarch', 'pentium3', 'pentium4', 'ppc32dy4', 'ppc64iseries', 'ppc64le', 'ppc64', 'ppc64p7', 'ppc64pseries', 'ppc8260', 'ppc8560', 'ppciseries', 'ppc', 'ppcpseries', 'riscv64', 's390', 's390x', 'sh3', 'sh4a', 'sh4', 'sh', 'sparc64', 'sparc64v', 'sparc', 'sparcv8', 'sparcv9', 'sparcv9v', 'x86_64']
    name, delimiter, arch = packagename.rpartition('.')
    if name and arch and (arch in redhat_rpm_arches):
        return (name, arch)
    return (packagename, None)