from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
class ElementSWDrive(object):
    """
    Element Software Storage Drive operations
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent', 'clean'], default='present'), drive_ids=dict(required=False, type='list', elements='str', aliases=['drive_id']), node_ids=dict(required=False, type='list', elements='str', aliases=['node_id']), force_during_upgrade=dict(required=False, type='bool'), force_during_bin_sync=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        input_params = self.module.params
        self.state = input_params['state']
        self.drive_ids = input_params['drive_ids']
        self.node_ids = input_params['node_ids']
        self.force_during_upgrade = input_params['force_during_upgrade']
        self.force_during_bin_sync = input_params['force_during_bin_sync']
        self.list_nodes = None
        self.debug = list()
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module, timeout=120)

    def get_node_id(self, node_id):
        """
            Get Node ID
            :description: Find and retrieve node_id from the active cluster

            :return: node_id (None if not found)
            :rtype: node_id
        """
        if self.list_nodes is None:
            self.list_nodes = self.sfe.list_active_nodes()
        for current_node in self.list_nodes.nodes:
            if node_id == str(current_node.node_id):
                return current_node.node_id
            elif node_id == current_node.name:
                return current_node.node_id
        self.module.fail_json(msg='unable to find node for node_id=%s' % node_id)

    def get_drives_listby_status(self, node_num_ids):
        """
            Capture list of drives based on status for a given node_id
            :description: Capture list of active, failed and available drives from a given node_id

            :return: None
        """
        self.active_drives = dict()
        self.available_drives = dict()
        self.other_drives = dict()
        self.all_drives = self.sfe.list_drives()
        for drive in self.all_drives.drives:
            if node_num_ids is None or drive.node_id in node_num_ids:
                if drive.status in ['active', 'failed']:
                    self.active_drives[drive.serial] = drive.drive_id
                elif drive.status == 'available':
                    self.available_drives[drive.serial] = drive.drive_id
                else:
                    self.other_drives[drive.serial] = (drive.drive_id, drive.status)
        self.debug.append('available: %s' % self.available_drives)
        self.debug.append('active: %s' % self.active_drives)
        self.debug.append('other: %s' % self.other_drives)

    def get_drive_id(self, drive_id, node_num_ids):
        """
            Get Drive ID
            :description: Find and retrieve drive_id from the active cluster
            Assumes self.all_drives is already populated

            :return: node_id (None if not found)
            :rtype: node_id
        """
        for drive in self.all_drives.drives:
            if drive_id == str(drive.drive_id):
                break
            if drive_id == drive.serial:
                break
        else:
            self.module.fail_json(msg='unable to find drive for drive_id=%s.  Debug=%s' % (drive_id, self.debug))
        if node_num_ids and drive.node_id not in node_num_ids:
            self.module.fail_json(msg='drive for drive_id=%s belongs to another node, with node_id=%d.  Debug=%s' % (drive_id, drive.node_id, self.debug))
        return (drive.drive_id, drive.status)

    def get_active_drives(self, drives):
        """
        return a list of active drives
        if drives is specified, only [] or a subset of disks in drives are returned
        else all available drives for this node or cluster are returned
        """
        if drives is None:
            return list(self.active_drives.values())
        return [drive_id for drive_id, status in drives if status in ['active', 'failed']]

    def get_available_drives(self, drives, action):
        """
        return a list of available drives (not active)
        if drives is specified, only [] or a subset of disks in drives are returned
        else all available drives for this node or cluster are returned
        """
        if drives is None:
            return list(self.available_drives.values())
        action_list = list()
        for drive_id, drive_status in drives:
            if drive_status == 'available':
                action_list.append(drive_id)
            elif drive_status in ['active', 'failed']:
                pass
            elif drive_status == 'erasing' and action == 'erase':
                pass
            elif drive_status == 'removing':
                self.module.fail_json(msg='Error - cannot %s drive while it is being removed.  Debug: %s' % (action, self.debug))
            elif drive_status == 'erasing' and action == 'add':
                self.module.fail_json(msg='Error - cannot %s drive while it is being erased.  Debug: %s' % (action, self.debug))
            else:
                self.module.fail_json(msg='Error - cannot %s drive while it is in %s state.  Debug: %s' % (action, drive_status, self.debug))
        return action_list

    def add_drive(self, drives=None):
        """
        Add Drive available for Cluster storage expansion
        """
        kwargs = dict()
        if self.force_during_upgrade is not None:
            kwargs['force_during_upgrade'] = self.force_during_upgrade
        if self.force_during_bin_sync is not None:
            kwargs['force_during_bin_sync'] = self.force_during_bin_sync
        try:
            self.sfe.add_drives(drives, **kwargs)
        except Exception as exception_object:
            self.module.fail_json(msg='Error adding drive%s: %s: %s' % ('s' if len(drives) > 1 else '', str(drives), to_native(exception_object)), exception=traceback.format_exc())

    def remove_drive(self, drives=None):
        """
        Remove Drive active in Cluster
        """
        kwargs = dict()
        if self.force_during_upgrade is not None:
            kwargs['force_during_upgrade'] = self.force_during_upgrade
        try:
            self.sfe.remove_drives(drives, **kwargs)
        except Exception as exception_object:
            self.module.fail_json(msg='Error removing drive%s: %s: %s' % ('s' if len(drives) > 1 else '', str(drives), to_native(exception_object)), exception=traceback.format_exc())

    def secure_erase(self, drives=None):
        """
        Secure Erase any residual data existing on a drive
        """
        try:
            self.sfe.secure_erase_drives(drives)
        except Exception as exception_object:
            self.module.fail_json(msg='Error cleaning data from drive%s: %s: %s' % ('s' if len(drives) > 1 else '', str(drives), to_native(exception_object)), exception=traceback.format_exc())

    def apply(self):
        """
        Check, process and initiate Drive operation
        """
        changed = False
        action_list = []
        node_num_ids = None
        drives = None
        if self.node_ids:
            node_num_ids = [self.get_node_id(node_id) for node_id in self.node_ids]
        self.get_drives_listby_status(node_num_ids)
        if self.drive_ids:
            drives = [self.get_drive_id(drive_id, node_num_ids) for drive_id in self.drive_ids]
        if self.state == 'present':
            action_list = self.get_available_drives(drives, 'add')
        elif self.state == 'absent':
            action_list = self.get_active_drives(drives)
        elif self.state == 'clean':
            action_list = self.get_available_drives(drives, 'erase')
        if len(action_list) > 0:
            changed = True
        if not self.module.check_mode and changed:
            if self.state == 'present':
                self.add_drive(action_list)
            elif self.state == 'absent':
                self.remove_drive(action_list)
            elif self.state == 'clean':
                self.secure_erase(action_list)
        self.module.exit_json(changed=changed)