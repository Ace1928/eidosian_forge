from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class PowerFlexReplicationPair(object):
    """Class with replication pair operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_replication_pair_parameters())
        mut_ex_args = [['rcg_name', 'rcg_id'], ['pair_id', 'pair_name']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True, mutually_exclusive=mut_ex_args)
        utils.ensure_required_libs(self.module)
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def get_replication_pair(self, pair_name=None, pair_id=None):
        """Get replication pair details
            :param pair_name: Name of the replication pair
            :param pair_id: ID of the replication pair
            :return: Replication pair details
        """
        name_or_id = pair_id if pair_id else pair_name
        try:
            pair_details = []
            if pair_id:
                pair_details = self.powerflex_conn.replication_pair.get(filter_fields={'id': pair_id})
            if pair_name:
                pair_details = self.powerflex_conn.replication_pair.get(filter_fields={'name': pair_name})
            if pair_details:
                pair_details[0].pop('links', None)
                pair_details[0]['localVolumeName'] = self.get_volume(pair_details[0]['localVolumeId'], filter_by_name=False)[0]['name']
                pair_details[0]['statistics'] = self.powerflex_conn.replication_pair.get_statistics(pair_details[0]['id'])
                return pair_details[0]
            return pair_details
        except Exception as e:
            errormsg = 'Failed to get the replication pair {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_volume(self, vol_name_or_id, filter_by_name=True, is_remote=False):
        """Get volume details
            :param vol_name: ID or name of the volume
            :param filter_by_name: If filter details by name or id
            :param is_remote: Specifies if source or target volume
            :return: Details of volume if exist.
        """
        try:
            volume_details = []
            filter_field = {'id': vol_name_or_id}
            if filter_by_name:
                filter_field = {'name': vol_name_or_id}
            if is_remote:
                self.remote_powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params['remote_peer'])
                volume_details = self.remote_powerflex_conn.volume.get(filter_fields=filter_field)
            else:
                volume_details = self.powerflex_conn.volume.get(filter_fields=filter_field)
            if not volume_details:
                vol_type = 'Target' if is_remote else 'Source'
                self.module.fail_json('%s volume %s does not exist' % (vol_type, vol_name_or_id))
            return volume_details
        except Exception as e:
            errormsg = 'Failed to retrieve volume {0}'.format(str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_rcg(self, rcg_name=None, rcg_id=None):
        """Get rcg details
            :param rcg_name: Name of the rcg
            :param rcg_id: ID of the rcg
            :return: RCG details
        """
        name_or_id = rcg_id if rcg_id else rcg_name
        try:
            rcg_details = {}
            if rcg_id:
                rcg_details = self.powerflex_conn.replication_consistency_group.get(filter_fields={'id': rcg_id})
            if rcg_name:
                rcg_details = self.powerflex_conn.replication_consistency_group.get(filter_fields={'name': rcg_name})
            if not rcg_details:
                self.module.fail_json('RCG %s does not exist' % rcg_name)
            return rcg_details[0]
        except Exception as e:
            errormsg = 'Failed to get the replication consistency group {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_rcg_replication_pairs(self, rcg_id):
        """Get rcg replication pair details
            :param rcg_id: ID of the rcg
            :return: RCG replication pair details
        """
        try:
            rcg_pairs = self.powerflex_conn.replication_consistency_group.get_replication_pairs(rcg_id)
            for rcg_pair in rcg_pairs:
                rcg_pair.pop('links', None)
                rcg_pair['localVolumeName'] = self.get_volume(rcg_pair['localVolumeId'], filter_by_name=False)[0]['name']
                rcg_pair['replicationConsistencyGroupName'] = self.get_rcg(rcg_id=rcg_pair['replicationConsistencyGroupId'])['name']
            return rcg_pairs
        except Exception as e:
            errormsg = 'Failed to get the replication pairs for replication consistency group {0} with error {1}'.format(rcg_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def create_replication_pairs(self, rcg_id, rcg_pairs, input_pairs):
        """Create replication pairs"""
        try:
            for pair in input_pairs:
                if pair['source_volume_name'] is not None:
                    pair['source_volume_id'] = self.get_volume(pair['source_volume_name'])[0]['id']
                if pair['target_volume_name'] is not None:
                    pair['target_volume_id'] = self.get_volume(pair['target_volume_name'], is_remote=True)[0]['id']
            pairs = find_non_existing_pairs(rcg_pairs, input_pairs)
            if not pairs:
                return False
            if not self.module.check_mode:
                for pair in pairs:
                    self.powerflex_conn.replication_pair.add(source_vol_id=pair['source_volume_id'], dest_vol_id=pair['target_volume_id'], rcg_id=rcg_id, copy_type=pair['copy_type'], name=pair['name'])
            return True
        except Exception as e:
            errormsg = 'Create replication pairs failed with error {0}'.format(str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def pause(self, pair_id):
        """Pause replication pair
            :param pair_id: ID of the replication pair
            :return: True if paused
        """
        try:
            if not self.module.check_mode:
                self.powerflex_conn.replication_pair.pause(pair_id)
            return True
        except Exception as e:
            errormsg = 'Pause replication pair {0} failed with error {1}'.format(pair_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def resume(self, pair_id):
        """Resume replication pair
            :param pair_id: ID of the replication pair
            :return: True if resumed
        """
        try:
            if not self.module.check_mode:
                self.powerflex_conn.replication_pair.resume(pair_id)
            return True
        except Exception as e:
            errormsg = 'Resume replication pair {0} failed with error {1}'.format(pair_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def delete_pair(self, pair_id):
        """Delete replication pair
            :param pair_id: Replication pair id.
            :return: Boolean indicates if delete pair operation is successful
        """
        try:
            if not self.module.check_mode:
                self.powerflex_conn.replication_pair.remove(pair_id)
            return True
        except Exception as e:
            errormsg = 'Delete replication pair {0} failed with error {1}'.format(pair_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_input(self, params):
        if params['pairs'] is not None:
            self.validate_pairs(params)
            if not params['rcg_id'] and (not params['rcg_name']):
                self.module.fail_json(msg='Specify either rcg_id or rcg_name to create replication pair')
        self.validate_pause(params)

    def validate_pairs(self, params):
        for pair in params['pairs']:
            if pair['source_volume_id'] and pair['source_volume_name']:
                self.module.fail_json(msg='Specify either source_volume_id or source_volume_name')
            if pair['target_volume_id'] and pair['target_volume_name']:
                self.module.fail_json(msg='Specify either target_volume_id or target_volume_name')
            if pair['target_volume_name'] and params['remote_peer'] is None:
                self.module.fail_json(msg='Specify remote_peer with target_volume_name')

    def validate_pause(self, params):
        if params['pause'] is not None and (not params['pair_id'] and (not params['pair_name'])):
            self.module.fail_json(msg='Specify either pair_id or pair_name to perform pause or resume of initial copy')

    def validate_pause_or_resume(self, pause, replication_pair_details, pair_id):
        if not replication_pair_details:
            self.module.fail_json(msg='Specify a valid pair_name or pair_id to perform pause or resume')
        return self.perform_pause_or_resume(pause, replication_pair_details, pair_id)

    def perform_pause_or_resume(self, pause, replication_pair_details, pair_id):
        changed = False
        if pause and replication_pair_details['initialCopyState'] not in ('Paused', 'Done'):
            changed = self.pause(pair_id)
        elif not pause and replication_pair_details['initialCopyState'] == 'Paused':
            changed = self.resume(pair_id)
        return changed

    def perform_module_operation(self):
        """
        Perform different actions on replication pair based on parameters passed in
        the playbook
        """
        self.validate_input(self.module.params)
        rcg_name = self.module.params['rcg_name']
        rcg_id = self.module.params['rcg_id']
        pair_name = self.module.params['pair_name']
        pair_id = self.module.params['pair_id']
        pairs = self.module.params['pairs']
        pause = self.module.params['pause']
        state = self.module.params['state']
        changed = False
        result = dict(changed=False, replication_pair_details=[], rcg_replication_pairs=[])
        if pair_id or pair_name:
            result['replication_pair_details'] = self.get_replication_pair(pair_name, pair_id)
            if result['replication_pair_details']:
                pair_id = result['replication_pair_details']['id']
        if pairs:
            rcg_id = self.get_rcg(rcg_name, rcg_id)['id']
            result['rcg_replication_pairs'] = self.get_rcg_replication_pairs(rcg_id)
            changed = self.create_replication_pairs(rcg_id, result['rcg_replication_pairs'], pairs)
            if changed:
                result['rcg_replication_pairs'] = self.get_rcg_replication_pairs(rcg_id)
        if pause is not None:
            changed = self.validate_pause_or_resume(pause, result['replication_pair_details'], pair_id)
        if state == 'absent' and result['replication_pair_details']:
            changed = self.delete_pair(pair_id)
        if changed and (pair_id or pair_name):
            result['replication_pair_details'] = self.get_replication_pair(pair_name, pair_id)
        result['changed'] = changed
        self.module.exit_json(**result)