from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _check_nameConstraints(extensions):
    current_nc_ext = _find_extension(extensions, cryptography.x509.NameConstraints)
    current_nc_perm = [to_text(altname) for altname in current_nc_ext.value.permitted_subtrees or []] if current_nc_ext else []
    current_nc_excl = [to_text(altname) for altname in current_nc_ext.value.excluded_subtrees or []] if current_nc_ext else []
    nc_perm = [to_text(cryptography_get_name(altname, 'name constraints permitted')) for altname in self.name_constraints_permitted]
    nc_excl = [to_text(cryptography_get_name(altname, 'name constraints excluded')) for altname in self.name_constraints_excluded]
    if set(nc_perm) != set(current_nc_perm) or set(nc_excl) != set(current_nc_excl):
        return False
    if nc_perm or nc_excl:
        if current_nc_ext.critical != self.name_constraints_critical:
            return False
    return True