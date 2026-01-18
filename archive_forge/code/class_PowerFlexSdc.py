from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
class PowerFlexSdc(object):
    """Class with SDC operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_sdc_parameters())
        mutually_exclusive = [['sdc_id', 'sdc_ip', 'sdc_name']]
        required_one_of = [['sdc_id', 'sdc_ip', 'sdc_name']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of)
        utils.ensure_required_libs(self.module)
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def rename_sdc(self, sdc_id, new_name):
        """Rename SDC
        :param sdc_id: The ID of the SDC
        :param new_name: The new name of the SDC
        :return: Boolean indicating if rename operation is successful
        """
        try:
            self.powerflex_conn.sdc.rename(sdc_id=sdc_id, name=new_name)
            return True
        except Exception as e:
            errormsg = 'Failed to rename SDC %s with error %s' % (sdc_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_mapped_volumes(self, sdc_id):
        """Get volumes mapped to SDC
        :param sdc_id: The ID of the SDC
        :return: List containing volume details mapped to SDC
        """
        try:
            resp = self.powerflex_conn.sdc.get_mapped_volumes(sdc_id=sdc_id)
            return resp
        except Exception as e:
            errormsg = 'Failed to get the volumes mapped to SDC %s with error %s' % (sdc_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_sdc(self, sdc_name=None, sdc_ip=None, sdc_id=None):
        """Get the SDC Details
            :param sdc_name: The name of the SDC
            :param sdc_ip: The IP of the SDC
            :param sdc_id: The ID of the SDC
            :return: The dict containing SDC details
        """
        if sdc_name:
            id_ip_name = sdc_name
        elif sdc_ip:
            id_ip_name = sdc_ip
        else:
            id_ip_name = sdc_id
        try:
            if sdc_name:
                sdc_details = self.powerflex_conn.sdc.get(filter_fields={'name': sdc_name})
            elif sdc_ip:
                sdc_details = self.powerflex_conn.sdc.get(filter_fields={'sdcIp': sdc_ip})
            else:
                sdc_details = self.powerflex_conn.sdc.get(filter_fields={'id': sdc_id})
            if len(sdc_details) == 0:
                error_msg = 'Unable to find SDC with identifier %s' % id_ip_name
                LOG.error(error_msg)
                return None
            sdc_details[0]['mapped_volumes'] = self.get_mapped_volumes(sdc_details[0]['id'])
            return sdc_details[0]
        except Exception as e:
            errormsg = 'Failed to get the SDC %s with error %s' % (id_ip_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_parameters(self, sdc_name=None, sdc_id=None, sdc_ip=None):
        """Validate the input parameters"""
        sdc_identifiers = ['sdc_name', 'sdc_id', 'sdc_ip']
        for param in sdc_identifiers:
            if self.module.params[param] is not None and len(self.module.params[param].strip()) == 0:
                msg = f'Please provide valid {param}'
                LOG.error(msg)
                self.module.fail_json(msg=msg)

    def remove(self, sdc_id):
        """Remove the SDC"""
        try:
            LOG.info(msg=f'Removing SDC {sdc_id}')
            self.powerflex_conn.sdc.delete(sdc_id)
            return True
        except Exception as e:
            errormsg = f'Removing SDC {sdc_id} failed with error {str(e)}'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def set_performance_profile(self, sdc_id, performance_profile):
        """Set performance profile of SDC"""
        try:
            LOG.info(msg=f'Setting performance profile of SDC {sdc_id}')
            self.powerflex_conn.sdc.set_performance_profile(sdc_id, performance_profile)
            return True
        except Exception as e:
            errormsg = f'Modifying performance profile of SDC {sdc_id} failed with error {str(e)}'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_input(self, sdc_details, sdc_new_name, state, id_ip_name):
        if state == 'present' and (not sdc_details):
            error_msg = 'Could not find any SDC instance with identifier %s.' % id_ip_name
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        if sdc_new_name and len(sdc_new_name.strip()) == 0:
            self.module.fail_json(msg='Provide valid SDC name to rename to.')

    def perform_modify(self, sdc_details, sdc_new_name, performance_profile):
        changed = False
        if sdc_new_name is not None and sdc_new_name != sdc_details['name']:
            changed = self.rename_sdc(sdc_details['id'], sdc_new_name)
        if performance_profile and performance_profile != sdc_details['perfProfile']:
            changed = self.set_performance_profile(sdc_details['id'], performance_profile)
        return changed

    def perform_module_operation(self):
        """
        Perform different actions on SDC based on parameters passed in
        the playbook
        """
        sdc_name = self.module.params['sdc_name']
        sdc_id = self.module.params['sdc_id']
        sdc_ip = self.module.params['sdc_ip']
        sdc_new_name = self.module.params['sdc_new_name']
        performance_profile = self.module.params['performance_profile']
        state = self.module.params['state']
        changed = False
        result = dict(changed=False, sdc_details={})
        self.validate_parameters(sdc_name, sdc_id, sdc_ip)
        sdc_details = self.get_sdc(sdc_name=sdc_name, sdc_id=sdc_id, sdc_ip=sdc_ip)
        id_ip_name = sdc_name or sdc_ip or sdc_id
        self.validate_input(sdc_details, sdc_new_name, state, id_ip_name)
        if state == 'absent' and sdc_details:
            changed = self.remove(sdc_details['id'])
        if state == 'present' and sdc_details:
            changed = self.perform_modify(sdc_details, sdc_new_name, performance_profile)
        if changed:
            sdc_details = self.get_sdc(sdc_name=sdc_new_name or sdc_name, sdc_id=sdc_id, sdc_ip=sdc_ip)
        result['sdc_details'] = sdc_details
        result['changed'] = changed
        self.module.exit_json(**result)