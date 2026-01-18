from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
class ImportLicense(License):
    STATUS_SUCCESS = [200, 202]

    def execute(self):
        """
        Executes the import license process based on the given module parameters.

        Args:
            module (object): The Ansible module object.

        Returns:
            object: The response object from the import license API call.
        """
        if not self.module.params.get('share_parameters').get('file_name'):
            self.module.exit_json(msg=MISSING_FILE_NAME_PARAMETER_MSG, failed=True)
        share_type = self.module.params.get('share_parameters').get('share_type')
        self.__check_file_extension()
        import_license_url = self.__get_import_license_url()
        resource_id = get_manager_res_id(self.idrac)
        job_status = {}
        if share_type == 'local':
            import_license_response = self.__import_license_local(import_license_url, resource_id)
        elif share_type in ['http', 'https']:
            import_license_response = self.__import_license_http(import_license_url, resource_id)
            job_status = self.get_job_status(import_license_response)
        elif share_type == 'cifs':
            import_license_response = self.__import_license_cifs(import_license_url, resource_id)
            job_status = self.get_job_status(import_license_response)
        elif share_type == 'nfs':
            import_license_response = self.__import_license_nfs(import_license_url, resource_id)
            job_status = self.get_job_status(import_license_response)
        status = import_license_response.status_code
        if status in self.STATUS_SUCCESS:
            self.module.exit_json(msg=SUCCESS_IMPORT_MSG, changed=True, job_details=job_status)
        else:
            self.module.exit_json(msg=FAILURE_IMPORT_MSG, failed=True, job_details=job_status)

    def __import_license_local(self, import_license_url, resource_id):
        """
        Import a license locally.

        Args:
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            dict: The import status of the license.
        """
        payload = {}
        path = self.module.params.get('share_parameters').get('share_name')
        if not (os.path.exists(path) or os.path.isdir(path)):
            self.module.exit_json(msg=INVALID_DIRECTORY_MSG.format(path=path), failed=True)
        file_path = self.module.params.get('share_parameters').get('share_name') + '/' + self.module.params.get('share_parameters').get('file_name')
        file_exits = os.path.exists(file_path)
        if file_exits:
            with open(file_path, 'rb') as cert:
                cert_content = cert.read()
                read_file = base64.encodebytes(cert_content).decode('ascii')
        else:
            self.module.exit_json(msg=NO_FILE_MSG, failed=True)
        payload['LicenseFile'] = read_file
        payload['FQDD'] = resource_id
        payload['ImportOptions'] = 'Force'
        try:
            import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
        except HTTPError as err:
            filter_err = remove_key(json.load(err), regex_pattern=ODATA_REGEX)
            message_details = filter_err.get('error').get('@Message.ExtendedInfo')[0]
            message_id = message_details.get('MessageId')
            if 'LIC018' in message_id:
                self.module.exit_json(msg=message_details.get('Message'), skipped=True)
            else:
                self.module.exit_json(msg=message_details.get('Message'), error_info=filter_err, failed=True)
        return import_status

    def __import_license_http(self, import_license_url, resource_id):
        """
        Imports a license using HTTP.

        Args:
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            object: The import status.
        """
        payload = {}
        payload['LicenseName'] = self.module.params.get('share_parameters').get('file_name')
        payload['FQDD'] = resource_id
        payload['ImportOptions'] = 'Force'
        proxy_details = self.get_proxy_details()
        payload.update(proxy_details)
        import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
        return import_status

    def __import_license_cifs(self, import_license_url, resource_id):
        """
        Imports a license using CIFS share type.

        Args:
            self (object): The instance of the class.
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            object: The import status of the license.
        """
        payload = {}
        payload['ShareType'] = 'CIFS'
        payload['LicenseName'] = self.module.params.get('share_parameters').get('file_name')
        payload['FQDD'] = resource_id
        payload['ImportOptions'] = 'Force'
        if self.module.params.get('share_parameters').get('workgroup'):
            payload['Workgroup'] = self.module.params.get('share_parameters').get('workgroup')
        share_details = self.get_share_details()
        payload.update(share_details)
        import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
        return import_status

    def __import_license_nfs(self, import_license_url, resource_id):
        """
        Import a license from an NFS share.

        Args:
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            dict: The import status of the license.
        """
        payload = {}
        payload['ShareType'] = 'NFS'
        payload['IPAddress'] = self.module.params.get('share_parameters').get('ip_address')
        payload['ShareName'] = self.module.params.get('share_parameters').get('share_name')
        payload['LicenseName'] = self.module.params.get('share_parameters').get('file_name')
        payload['FQDD'] = resource_id
        payload['ImportOptions'] = 'Force'
        import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
        return import_status

    def __check_file_extension(self):
        """
        Check if the file extension of the given file name is valid.

        :param module: The Ansible module object.
        :type module: AnsibleModule

        :return: None
        """
        share_type = self.module.params.get('share_parameters').get('share_type')
        file_name = self.module.params.get('share_parameters').get('file_name')
        valid_extensions = {'.txt', '.xml'} if share_type == 'local' else {'.xml'}
        file_extension = any((file_name.lower().endswith(ext) for ext in valid_extensions))
        if not file_extension:
            self.module.exit_json(msg=INVALID_FILE_MSG, failed=True)

    def __get_import_license_url(self):
        """
        Get the import license URL.

        :param module: The module object.
        :type module: object
        :return: The import license URL.
        :rtype: str
        """
        uri, error_msg = validate_and_get_first_resource_id_uri(self.module, self.idrac, MANAGERS_URI)
        if error_msg:
            self.module.exit_json(msg=error_msg, failed=True)
        resp = get_dynamic_uri(self.idrac, uri)
        url = resp.get('Links', {}).get(OEM, {}).get(MANUFACTURER, {}).get(LICENSE_MANAGEMENT_SERVICE, {}).get(ODATA, {})
        action_resp = get_dynamic_uri(self.idrac, url)
        license_service = IMPORT_LOCAL if self.module.params.get('share_parameters').get('share_type') == 'local' else IMPORT_NETWORK_SHARE
        import_url = action_resp.get(ACTIONS, {}).get(license_service, {}).get('target', {})
        return import_url

    def get_job_status(self, license_job_response):
        res_uri = validate_and_get_first_resource_id_uri(self.module, self.idrac, MANAGERS_URI)
        job_tracking_uri = license_job_response.headers.get('Location')
        job_id = job_tracking_uri.split('/')[-1]
        job_uri = IDRAC_JOB_URI.format(job_id=job_id, res_uri=res_uri[0])
        job_failed, msg, job_dict, wait_time = idrac_redfish_job_tracking(self.idrac, job_uri)
        job_dict = remove_key(job_dict, regex_pattern=ODATA_REGEX)
        if job_failed:
            if job_dict.get('MessageId') == 'LIC018':
                self.module.exit_json(msg=job_dict.get('Message'), skipped=True, job_details=job_dict)
            else:
                self.module.exit_json(msg=job_dict.get('Message'), failed=True, job_details=job_dict)
        return job_dict