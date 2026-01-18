from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
class AnsibleFortios(object):

    def __init__(self, module):
        if not HAS_PYFG:
            module.fail_json(msg='Could not import the python library pyFG required by this module')
        self.result = {'changed': False}
        self.module = module

    def _connect(self):
        if self.module.params['file_mode']:
            self.forti_device = FortiOS('')
        else:
            host = self.module.params['host']
            username = self.module.params['username']
            password = self.module.params['password']
            timeout = self.module.params['timeout']
            vdom = self.module.params['vdom']
            self.forti_device = FortiOS(host, username=username, password=password, timeout=timeout, vdom=vdom)
            try:
                self.forti_device.open()
            except Exception as e:
                self.module.fail_json(msg='Error connecting device. %s' % to_text(e), exception=traceback.format_exc())

    def load_config(self, path):
        self.path = path
        self._connect()
        if self.module.params['file_mode']:
            try:
                f = open(self.module.params['config_file'], 'r')
                running = f.read()
                f.close()
            except IOError as e:
                self.module.fail_json(msg='Error reading configuration file. %s' % to_text(e), exception=traceback.format_exc())
            self.forti_device.load_config(config_text=running, path=path)
        else:
            try:
                self.forti_device.load_config(path=path)
            except Exception as e:
                self.forti_device.close()
                self.module.fail_json(msg='Error reading running config. %s' % to_text(e), exception=traceback.format_exc())
        self.result['running_config'] = self.forti_device.running_config.to_text()
        self.candidate_config = self.forti_device.candidate_config
        if self.module.params['backup']:
            backup(self.module, self.forti_device.running_config.to_text())

    def apply_changes(self):
        change_string = self.forti_device.compare_config()
        if change_string:
            self.result['change_string'] = change_string
            self.result['changed'] = True
        if change_string and (not self.module.check_mode):
            if self.module.params['file_mode']:
                try:
                    f = open(self.module.params['config_file'], 'w')
                    f.write(self.candidate_config.to_text())
                    f.close()
                except IOError as e:
                    self.module.fail_json(msg='Error writing configuration file. %s' % to_text(e), exception=traceback.format_exc())
            else:
                try:
                    self.forti_device.commit()
                except FailedCommit as e:
                    self.forti_device.close()
                    error_list = self.get_error_infos(e)
                    self.module.fail_json(msg_error_list=error_list, msg='Unable to commit change, check your args, the error was %s' % e.message)
                self.forti_device.close()
        self.module.exit_json(**self.result)

    def del_block(self, block_id):
        self.forti_device.candidate_config[self.path].del_block(block_id)

    def add_block(self, block_id, block):
        self.forti_device.candidate_config[self.path][block_id] = block

    def get_error_infos(self, cli_errors):
        error_list = []
        for errors in cli_errors.args:
            for error in errors:
                error_code = error[0]
                error_string = error[1]
                error_type = fortios_error_codes.get(error_code, 'unknown')
                error_list.append(dict(error_code=error_code, error_type=error_type, error_string=error_string))
        return error_list

    def get_empty_configuration_block(self, block_name, block_type):
        return FortiConfig(block_name, block_type)