from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def CheckAssociatedStorageProfile(self, profileManager, ref, name):
    """
        Check the associated storage policy profile.

        :param profileManager: A VMware Storage Policy Service manager object.
        :type profileManager: pbm.profile.ProfileManager
        :param ref: A server object ref to a virtual machine, virtual disk,
            or datastore.
        :type ref: pbm.ServerObjectRef
        :param name: A VMware storage policy profile name.
        :type name: str
        :returns: True if storage policy profile by name is associated to ref.
        :rtype: bool
        """
    profileIds = profileManager.PbmQueryAssociatedProfile(ref)
    if len(profileIds) > 0:
        profiles = profileManager.PbmRetrieveContent(profileIds=profileIds)
        for profile in profiles:
            if profile.name == name:
                return True
    return False