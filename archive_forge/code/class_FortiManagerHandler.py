from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGR_RC
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGBaseException
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGRCommon
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import scrub_dict
class FortiManagerHandler(object):

    def __init__(self, conn, module):
        self._conn = conn
        self._module = module
        self._tools = FMGRCommon

    def process_request(self, url, datagram, method):
        """
        Formats and Runs the API Request via Connection Plugin. Streamlined for use FROM Modules.

        :param url: Connection URL to access
        :type url: string
        :param datagram: The prepared payload for the API Request in dictionary format
        :type datagram: dict
        :param method: The preferred API Request method (GET, ADD, POST, etc....)
        :type method: basestring

        :return: Dictionary containing results of the API Request via Connection Plugin
        :rtype: dict
        """
        data = self._tools.format_request(method, url, **datagram)
        response = self._conn.send_request(method, data)
        return response

    def govern_response(self, module, results, msg=None, good_codes=None, stop_on_fail=None, stop_on_success=None, skipped=None, changed=None, unreachable=None, failed=None, success=None, changed_if_success=None, ansible_facts=None):
        """
        This function will attempt to apply default values to canned responses from FortiManager we know of.
        This saves time, and turns the response in the module into a "one-liner", while still giving us...
        the flexibility to directly use return_response in modules if we have too. This function saves repeated code.

        :param module: The Ansible Module CLASS object, used to run fail/exit json
        :type module: object
        :param msg: An overridable custom message from the module that called this.
        :type msg: string
        :param results: A dictionary object containing an API call results
        :type results: dict
        :param good_codes: A list of exit codes considered successful from FortiManager
        :type good_codes: list
        :param stop_on_fail: If true, stops playbook run when return code is NOT IN good codes (default: true)
        :type stop_on_fail: boolean
        :param stop_on_success: If true, stops playbook run when return code is IN good codes (default: false)
        :type stop_on_success: boolean
        :param changed: If True, tells Ansible that object was changed (default: false)
        :type skipped: boolean
        :param skipped: If True, tells Ansible that object was skipped (default: false)
        :type skipped: boolean
        :param unreachable: If True, tells Ansible that object was unreachable (default: false)
        :type unreachable: boolean
        :param failed: If True, tells Ansible that execution was a failure. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param success: If True, tells Ansible that execution was a success. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param changed_if_success: If True, defaults to changed if successful if you specify or not"
        :type changed_if_success: boolean
        :param ansible_facts: A prepared dictionary of ansible facts from the execution.
        :type ansible_facts: dict
        """
        if module is None and results is None:
            raise FMGBaseException('govern_response() was called without a module and/or results tuple! Fix!')
        try:
            rc = results[0]
        except BaseException:
            raise FMGBaseException('govern_response() was called without the return code at results[0]')
        rc_data = None
        try:
            rc_codes = FMGR_RC.get('fmgr_return_codes')
            rc_data = rc_codes.get(rc)
        except BaseException:
            pass
        if not rc_data:
            rc_data = {}
        if good_codes is not None:
            rc_data['good_codes'] = good_codes
        if stop_on_fail is not None:
            rc_data['stop_on_fail'] = stop_on_fail
        if stop_on_success is not None:
            rc_data['stop_on_success'] = stop_on_success
        if skipped is not None:
            rc_data['skipped'] = skipped
        if changed is not None:
            rc_data['changed'] = changed
        if unreachable is not None:
            rc_data['unreachable'] = unreachable
        if failed is not None:
            rc_data['failed'] = failed
        if success is not None:
            rc_data['success'] = success
        if changed_if_success is not None:
            rc_data['changed_if_success'] = changed_if_success
        if results is not None:
            rc_data['results'] = results
        if msg is not None:
            rc_data['msg'] = msg
        if ansible_facts is None:
            rc_data['ansible_facts'] = {}
        else:
            rc_data['ansible_facts'] = ansible_facts
        return self.return_response(module=module, results=results, msg=rc_data.get('msg', 'NULL'), good_codes=rc_data.get('good_codes', (0,)), stop_on_fail=rc_data.get('stop_on_fail', True), stop_on_success=rc_data.get('stop_on_success', False), skipped=rc_data.get('skipped', False), changed=rc_data.get('changed', False), changed_if_success=rc_data.get('changed_if_success', False), unreachable=rc_data.get('unreachable', False), failed=rc_data.get('failed', False), success=rc_data.get('success', False), ansible_facts=rc_data.get('ansible_facts', dict()))

    @staticmethod
    def return_response(module, results, msg='NULL', good_codes=(0,), stop_on_fail=True, stop_on_success=False, skipped=False, changed=False, unreachable=False, failed=False, success=False, changed_if_success=True, ansible_facts=()):
        """
        This function controls the logout and error reporting after an method or function runs. The exit_json for
        ansible comes from logic within this function. If this function returns just the msg, it means to continue
        execution on the playbook. It is called from the ansible module, or from the self.govern_response function.

        :param module: The Ansible Module CLASS object, used to run fail/exit json
        :type module: object
        :param msg: An overridable custom message from the module that called this.
        :type msg: string
        :param results: A dictionary object containing an API call results
        :type results: dict
        :param good_codes: A list of exit codes considered successful from FortiManager
        :type good_codes: list
        :param stop_on_fail: If true, stops playbook run when return code is NOT IN good codes (default: true)
        :type stop_on_fail: boolean
        :param stop_on_success: If true, stops playbook run when return code is IN good codes (default: false)
        :type stop_on_success: boolean
        :param changed: If True, tells Ansible that object was changed (default: false)
        :type skipped: boolean
        :param skipped: If True, tells Ansible that object was skipped (default: false)
        :type skipped: boolean
        :param unreachable: If True, tells Ansible that object was unreachable (default: false)
        :type unreachable: boolean
        :param failed: If True, tells Ansible that execution was a failure. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param success: If True, tells Ansible that execution was a success. Overrides good_codes. (default: false)
        :type unreachable: boolean
        :param changed_if_success: If True, defaults to changed if successful if you specify or not"
        :type changed_if_success: boolean
        :param ansible_facts: A prepared dictionary of ansible facts from the execution.
        :type ansible_facts: dict

        :return: A string object that contains an error message
        :rtype: str
        """
        if len(results) == 0 or (failed and success) or (changed and unreachable):
            module.exit_json(msg='Handle_response was called with no results, or conflicting failed/success or changed/unreachable parameters. Fix the exit code on module. Generic Failure', failed=True)
        if not failed and (not success):
            if len(results) > 0:
                if results[0] not in good_codes:
                    failed = True
                elif results[0] in good_codes:
                    success = True
        if len(results) > 0:
            if msg == 'NULL':
                try:
                    msg = results[1]['status']['message']
                except BaseException:
                    msg = 'No status message returned at results[1][status][message], and none supplied to msg parameter for handle_response.'
            if failed:
                if failed and skipped:
                    failed = False
                if failed and unreachable:
                    failed = False
                if stop_on_fail:
                    module.exit_json(msg=msg, failed=failed, changed=changed, unreachable=unreachable, skipped=skipped, results=results[1], ansible_facts=ansible_facts, rc=results[0], invocation={'module_args': ansible_facts['ansible_params']})
            elif success:
                if changed_if_success:
                    changed = True
                    success = False
                if stop_on_success:
                    module.exit_json(msg=msg, success=success, changed=changed, unreachable=unreachable, skipped=skipped, results=results[1], ansible_facts=ansible_facts, rc=results[0], invocation={'module_args': ansible_facts['ansible_params']})
        return msg

    def construct_ansible_facts(self, response, ansible_params, paramgram, *args, **kwargs):
        """
        Constructs a dictionary to return to ansible facts, containing various information about the execution.

        :param response: Contains the response from the FortiManager.
        :type response: dict
        :param ansible_params: Contains the parameters Ansible was called with.
        :type ansible_params: dict
        :param paramgram: Contains the paramgram passed to the modules' local modify function.
        :type paramgram: dict
        :param args: Free-form arguments that could be added.
        :param kwargs: Free-form keyword arguments that could be added.

        :return: A dictionary containing lots of information to append to Ansible Facts.
        :rtype: dict
        """
        facts = {'response': response, 'ansible_params': scrub_dict(ansible_params), 'paramgram': scrub_dict(paramgram), 'connected_fmgr': self._conn.return_connected_fmgr()}
        if args:
            facts['custom_args'] = args
        if kwargs:
            facts.update(kwargs)
        return facts