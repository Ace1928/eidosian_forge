from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
class DnacBase:
    """Class contains members which can be reused for all intent modules"""
    __metaclass__ = ABCMeta
    __is_log_init = False

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.config = copy.deepcopy(module.params.get('config'))
        self.have = {}
        self.want = {}
        self.validated_config = []
        self.msg = ''
        self.status = 'success'
        dnac_params = self.get_dnac_params(self.params)
        self.dnac = DNACSDK(params=dnac_params)
        self.dnac_apply = {'exec': self.dnac._exec}
        self.get_diff_state_apply = {'merged': self.get_diff_merged, 'deleted': self.get_diff_deleted, 'replaced': self.get_diff_replaced, 'overridden': self.get_diff_overridden, 'gathered': self.get_diff_gathered, 'rendered': self.get_diff_rendered, 'parsed': self.get_diff_parsed}
        self.verify_diff_state_apply = {'merged': self.verify_diff_merged, 'deleted': self.verify_diff_deleted, 'replaced': self.verify_diff_replaced, 'overridden': self.verify_diff_overridden, 'gathered': self.verify_diff_gathered, 'rendered': self.verify_diff_rendered, 'parsed': self.verify_diff_parsed}
        self.dnac_log = dnac_params.get('dnac_log')
        if self.dnac_log and (not DnacBase.__is_log_init):
            self.dnac_log_level = dnac_params.get('dnac_log_level') or 'WARNING'
            self.dnac_log_level = self.dnac_log_level.upper()
            self.validate_dnac_log_level()
            self.dnac_log_file_path = dnac_params.get('dnac_log_file_path') or 'dnac.log'
            self.validate_dnac_log_file_path()
            self.dnac_log_mode = 'w' if not dnac_params.get('dnac_log_append') else 'a'
            self.setup_logger('logger')
            self.logger = logging.getLogger('logger')
            DnacBase.__is_log_init = True
            self.log('Logging configured and initiated', 'DEBUG')
        elif not self.dnac_log:
            self.logger = logging.getLogger('empty_logger')
        self.log('Cisco Catalyst Center parameters: {0}'.format(dnac_params), 'DEBUG')
        self.supported_states = ['merged', 'deleted', 'replaced', 'overridden', 'gathered', 'rendered', 'parsed']
        self.result = {'changed': False, 'diff': [], 'response': [], 'warnings': []}

    @abstractmethod
    def validate_input(self):
        if not self.config:
            self.msg = 'config not available in playbook for validation'
            self.status = 'failed'
            return self

    def get_diff_merged(self):
        self.merged = True
        return self

    def get_diff_deleted(self):
        self.deleted = True
        return self

    def get_diff_replaced(self):
        self.replaced = True
        return self

    def get_diff_overridden(self):
        self.overridden = True
        return self

    def get_diff_gathered(self):
        self.gathered = True
        return self

    def get_diff_rendered(self):
        self.rendered = True
        return self

    def get_diff_parsed(self):
        self.parsed = True
        return self

    def verify_diff_merged(self):
        self.merged = True
        return self

    def verify_diff_deleted(self):
        self.deleted = True
        return self

    def verify_diff_replaced(self):
        self.replaced = True
        return self

    def verify_diff_overridden(self):
        self.overridden = True
        return self

    def verify_diff_gathered(self):
        self.gathered = True
        return self

    def verify_diff_rendered(self):
        self.rendered = True
        return self

    def verify_diff_parsed(self):
        self.parsed = True
        return self

    def setup_logger(self, logger_name):
        """Set up a logger with specified name and configuration based on dnac_log_level"""
        level_mapping = {'INFO': logging.INFO, 'DEBUG': logging.DEBUG, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}
        level = level_mapping.get(self.dnac_log_level, logging.WARNING)
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%m-%d-%Y %H:%M:%S')
        file_handler = logging.FileHandler(self.dnac_log_file_path, mode=self.dnac_log_mode)
        file_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(file_handler)

    def validate_dnac_log_level(self):
        """Validates if the logging level is string and of expected value"""
        if self.dnac_log_level not in ('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'):
            raise ValueError("Invalid log level: 'dnac_log_level:{0}'".format(self.dnac_log_level))

    def validate_dnac_log_file_path(self):
        """
        Validates the specified log file path, ensuring it is either absolute or relative,
        the directory exists, and has a .log extension.
        """
        dnac_log_file_path = os.path.abspath(self.dnac_log_file_path)
        log_directory = os.path.dirname(dnac_log_file_path)
        if not os.path.exists(log_directory):
            raise FileNotFoundError("The directory for log file '{0}' does not exist.".format(dnac_log_file_path))

    def log(self, message, level='WARNING', frameIncrement=0):
        """Logs formatted messages with specified log level and incrementing the call stack frame
        Args:
            self (obj, required): An instance of the DnacBase Class.
            message (str, required): The log message to be recorded.
            level (str, optional): The log level, default is "info".
                                   The log level can be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        """
        if self.dnac_log:
            class_name = self.__class__.__name__
            callerframerecord = inspect.stack()[1 + frameIncrement]
            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)
            log_message = ' %s: %s: %s: %s \n' % (class_name, info.function, info.lineno, message)
            log_method = getattr(self.logger, level.lower())
            log_method(log_message)

    def check_return_status(self):
        """API to check the return status value and exit/fail the module"""
        self.log('status: {0}, msg: {1}'.format(self.status, self.msg), 'DEBUG')
        if 'failed' in self.status:
            self.module.fail_json(msg=self.msg, response=[])
        elif 'exited' in self.status:
            self.module.exit_json(**self.result)
        elif 'invalid' in self.status:
            self.module.fail_json(msg=self.msg, response=[])

    def is_valid_password(self, password):
        """
        Check if a password is valid.
        Args:
            self (object): An instance of a class that provides access to Cisco Catalyst Center.
            password (str): The password to be validated.
        Returns:
            bool: True if the password is valid, False otherwise.
        Description:
            The function checks the validity of a password based on the following criteria:
            - Minimum 8 characters.
            - At least one lowercase letter.
            - At least one uppercase letter.
            - At least one digit.
            - At least one special character
        """
        pattern = '^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[-=\\\\;,./~!@#$%^&*()_+{}[\\]|:?]).{8,}$'
        return re.match(pattern, password) is not None

    def get_dnac_params(self, params):
        """Store the Cisco Catalyst Center parameters from the playbook"""
        dnac_params = {'dnac_host': params.get('dnac_host'), 'dnac_port': params.get('dnac_port'), 'dnac_username': params.get('dnac_username'), 'dnac_password': params.get('dnac_password'), 'dnac_verify': params.get('dnac_verify'), 'dnac_debug': params.get('dnac_debug'), 'dnac_log': params.get('dnac_log'), 'dnac_log_level': params.get('dnac_log_level'), 'dnac_log_file_path': params.get('dnac_log_file_path'), 'dnac_log_append': params.get('dnac_log_append')}
        return dnac_params

    def get_task_details(self, task_id):
        """
        Get the details of a specific task in Cisco Catalyst Center.
        Args:
            self (object): An instance of a class that provides access to Cisco Catalyst Center.
            task_id (str): The unique identifier of the task for which you want to retrieve details.
        Returns:
            dict or None: A dictionary containing detailed information about the specified task,
            or None if the task with the given task_id is not found.
        Description:
            If the task with the specified task ID is not found in Cisco Catalyst Center, this function will return None.
        """
        result = None
        response = self.dnac._exec(family='task', function='get_task_by_id', params={'task_id': task_id})
        self.log('Task Details: {0}'.format(str(response)), 'DEBUG')
        self.log("Retrieving task details by the API 'get_task_by_id' using task ID: {0}, Response: {1}".format(task_id, response), 'DEBUG')
        if response and isinstance(response, dict):
            result = response.get('response')
        return result

    def check_task_response_status(self, response, validation_string, data=False):
        """
        Get the site id from the site name.

        Parameters:
            self - The current object details.
            response (dict) - API response.
            validation_string (string) - String used to match the progress status.

        Returns:
            self
        """
        if not response:
            self.msg = 'response is empty'
            self.status = 'exited'
            return self
        if not isinstance(response, dict):
            self.msg = 'response is not a dictionary'
            self.status = 'exited'
            return self
        response = response.get('response')
        if response.get('errorcode') is not None:
            self.msg = response.get('response').get('detail')
            self.status = 'failed'
            return self
        task_id = response.get('taskId')
        while True:
            task_details = self.get_task_details(task_id)
            self.log('Getting task details from task ID {0}: {1}'.format(task_id, task_details), 'DEBUG')
            if task_details.get('isError') is True:
                if task_details.get('failureReason'):
                    self.msg = str(task_details.get('failureReason'))
                else:
                    self.msg = str(task_details.get('progress'))
                self.status = 'failed'
                break
            if validation_string in task_details.get('progress').lower():
                self.result['changed'] = True
                if data is True:
                    self.msg = task_details.get('data')
                self.status = 'success'
                break
            self.log('progress set to {0} for taskid: {1}'.format(task_details.get('progress'), task_id), 'DEBUG')
        return self

    def reset_values(self):
        """Reset all neccessary attributes to default values"""
        self.have.clear()
        self.want.clear()

    def get_execution_details(self, execid):
        """
        Get the execution details of an API

        Parameters:
            execid (str) - Id for API execution

        Returns:
            response (dict) - Status for API execution
        """
        self.log('Execution Id: {0}'.format(execid), 'DEBUG')
        response = self.dnac._exec(family='task', function='get_business_api_execution_details', params={'execution_id': execid})
        self.log('Response for the current execution: {0}'.format(response))
        return response

    def check_execution_response_status(self, response):
        """
        Checks the reponse status provided by API in the Cisco Catalyst Center

        Parameters:
            response (dict) - API response

        Returns:
            self
        """
        if not response:
            self.msg = 'response is empty'
            self.status = 'failed'
            return self
        if not isinstance(response, dict):
            self.msg = 'response is not a dictionary'
            self.status = 'failed'
            return self
        executionid = response.get('executionId')
        while True:
            execution_details = self.get_execution_details(executionid)
            if execution_details.get('status') == 'SUCCESS':
                self.result['changed'] = True
                self.msg = 'Successfully executed'
                self.status = 'success'
                break
            if execution_details.get('bapiError'):
                self.msg = execution_details.get('bapiError')
                self.status = 'failed'
                break
        return self

    def check_string_dictionary(self, task_details_data):
        """
        Check whether the input is string dictionary or string.

        Parameters:
            task_details_data (string) - Input either string dictionary or string.

        Returns:
            value (dict) - If the input is string dictionary, else returns None.
        """
        try:
            value = json.loads(task_details_data)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        return None

    def camel_to_snake_case(self, config):
        """
        Convert camel case keys to snake case keys in the config.

        Parameters:
            config (list) - Playbook details provided by the user.

        Returns:
            new_config (list) - Updated config after eliminating the camel cases.
        """
        if isinstance(config, dict):
            new_config = {}
            for key, value in config.items():
                new_key = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', key).lower()
                if new_key != key:
                    self.log('{0} will be deprecated soon. Please use {1}.'.format(key, new_key), 'DEBUG')
                new_value = self.camel_to_snake_case(value)
                new_config[new_key] = new_value
        elif isinstance(config, list):
            return [self.camel_to_snake_case(item) for item in config]
        else:
            return config
        return new_config

    def update_site_type_key(self, config):
        """
        Replace 'site_type' key with 'type' in the config.

        Parameters:
            config (list or dict) - Configuration details.

        Returns:
            updated_config (list or dict) - Updated config after replacing the keys.
        """
        if isinstance(config, dict):
            new_config = {}
            for key, value in config.items():
                if key == 'site_type':
                    new_key = 'type'
                else:
                    new_key = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', key).lower()
                new_value = self.update_site_type_key(value)
                new_config[new_key] = new_value
        elif isinstance(config, list):
            return [self.update_site_type_key(item) for item in config]
        else:
            return config
        return new_config