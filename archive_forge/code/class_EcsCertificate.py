from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
class EcsCertificate(object):
    """
    Entrust Certificate Services certificate class.
    """

    def __init__(self, module):
        self.path = module.params['path']
        self.full_chain_path = module.params['full_chain_path']
        self.force = module.params['force']
        self.backup = module.params['backup']
        self.request_type = module.params['request_type']
        self.csr = module.params['csr']
        self.changed = False
        self.filename = None
        self.tracking_id = None
        self.cert_status = None
        self.serial_number = None
        self.cert_days = None
        self.cert_details = None
        self.backup_file = None
        self.backup_full_chain_file = None
        self.cert = None
        self.ecs_client = None
        if self.path and os.path.exists(self.path):
            try:
                self.cert = load_certificate(self.path, backend='cryptography')
            except Exception as dummy:
                self.cert = None
        try:
            self.ecs_client = ECSClient(entrust_api_user=module.params['entrust_api_user'], entrust_api_key=module.params['entrust_api_key'], entrust_api_cert=module.params['entrust_api_client_cert_path'], entrust_api_cert_key=module.params['entrust_api_client_cert_key_path'], entrust_api_specification_path=module.params['entrust_api_specification_path'])
        except SessionConfigurationException as e:
            module.fail_json(msg='Failed to initialize Entrust Provider: {0}'.format(to_native(e)))
        try:
            self.ecs_client.GetAppVersion()
        except RestOperationException as e:
            module.fail_json(msg='Please verify credential information. Received exception when testing ECS connection: {0}'.format(to_native(e.message)))

    def convert_tracking_params(self, module):
        body = {}
        tracking = {}
        if module.params['requester_name']:
            tracking['requesterName'] = module.params['requester_name']
        if module.params['requester_email']:
            tracking['requesterEmail'] = module.params['requester_email']
        if module.params['requester_phone']:
            tracking['requesterPhone'] = module.params['requester_phone']
        if module.params['tracking_info']:
            tracking['trackingInfo'] = module.params['tracking_info']
        if module.params['custom_fields']:
            custom_fields = {}
            for k, v in module.params['custom_fields'].items():
                if v is not None:
                    custom_fields[k] = v
            tracking['customFields'] = custom_fields
        if module.params['additional_emails']:
            tracking['additionalEmails'] = module.params['additional_emails']
        body['tracking'] = tracking
        return body

    def convert_cert_subject_params(self, module):
        body = {}
        if module.params['subject_alt_name']:
            body['subjectAltName'] = module.params['subject_alt_name']
        if module.params['org']:
            body['org'] = module.params['org']
        if module.params['ou']:
            body['ou'] = module.params['ou']
        return body

    def convert_general_params(self, module):
        body = {}
        if module.params['eku']:
            body['eku'] = module.params['eku']
        if self.request_type == 'new':
            body['certType'] = module.params['cert_type']
        body['clientId'] = module.params['client_id']
        body.update(convert_module_param_to_json_bool(module, 'ctLog', 'ct_log'))
        body.update(convert_module_param_to_json_bool(module, 'endUserKeyStorageAgreement', 'end_user_key_storage_agreement'))
        return body

    def convert_expiry_params(self, module):
        body = {}
        if module.params['cert_lifetime']:
            body['certLifetime'] = module.params['cert_lifetime']
        elif module.params['cert_expiry']:
            body['certExpiryDate'] = module.params['cert_expiry']
        elif self.request_type != 'reissue':
            gmt_now = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
            expiry = gmt_now + datetime.timedelta(days=365)
            body['certExpiryDate'] = expiry.strftime('%Y-%m-%dT%H:%M:%S.00Z')
        return body

    def set_tracking_id_by_serial_number(self, module):
        try:
            serial_number = '{0:X}'.format(self.cert.serial_number)
            cert_results = self.ecs_client.GetCertificates(serialNumber=serial_number).get('certificates', {})
            if len(cert_results) == 1:
                self.tracking_id = cert_results[0].get('trackingId')
        except RestOperationException as dummy:
            return

    def set_cert_details(self, module):
        try:
            self.cert_details = self.ecs_client.GetCertificate(trackingId=self.tracking_id)
            self.cert_status = self.cert_details.get('status')
            self.serial_number = self.cert_details.get('serialNumber')
            self.cert_days = calculate_cert_days(self.cert_details.get('expiresAfter'))
        except RestOperationException as e:
            module.fail_json('Failed to get details of certificate with tracking_id="{0}", Error: '.format(self.tracking_id), to_native(e.message))

    def check(self, module):
        if self.cert:
            self.set_tracking_id_by_serial_number(module)
            if module.params['tracking_id'] and self.tracking_id and (module.params['tracking_id'] != self.tracking_id):
                module.warn('tracking_id parameter of "{0}" provided, but will be ignored. Valid certificate was present in path "{1}" with tracking_id of "{2}".'.format(module.params['tracking_id'], self.path, self.tracking_id))
        if not self.tracking_id:
            self.tracking_id = module.params['tracking_id']
        if not self.tracking_id:
            return False
        self.set_cert_details(module)
        if self.cert_status == 'EXPIRED' or self.cert_status == 'SUSPENDED' or self.cert_status == 'REVOKED':
            return False
        if self.cert_days < module.params['remaining_days']:
            return False
        return True

    def request_cert(self, module):
        if not self.check(module) or self.force:
            body = {}
            if self.csr and os.path.exists(self.csr):
                with open(self.csr, 'r') as csr_file:
                    body['csr'] = csr_file.read()
            if self.request_type != 'new' and (not self.tracking_id):
                module.warn('No existing Entrust certificate found in path={0} and no tracking_id was provided, setting request_type to "new" for this taskrun. Future playbook runs that point to the pathination file in {1} will use request_type={2}'.format(self.path, self.path, self.request_type))
                self.request_type = 'new'
            elif self.request_type == 'new' and self.tracking_id:
                module.warn('Existing certificate being acted upon, but request_type is "new", so will be a new certificate issuance rather than areissue or renew')
            body.update(self.convert_tracking_params(module))
            body.update(self.convert_cert_subject_params(module))
            body.update(self.convert_general_params(module))
            body.update(self.convert_expiry_params(module))
            if not module.check_mode:
                try:
                    if self.request_type == 'validate_only':
                        body['validateOnly'] = 'true'
                        result = self.ecs_client.NewCertRequest(Body=body)
                    if self.request_type == 'new':
                        result = self.ecs_client.NewCertRequest(Body=body)
                    elif self.request_type == 'renew':
                        result = self.ecs_client.RenewCertRequest(trackingId=self.tracking_id, Body=body)
                    elif self.request_type == 'reissue':
                        result = self.ecs_client.ReissueCertRequest(trackingId=self.tracking_id, Body=body)
                    self.tracking_id = result.get('trackingId')
                    self.set_cert_details(module)
                except RestOperationException as e:
                    module.fail_json(msg='Failed to request new certificate from Entrust (ECS) {0}'.format(e.message))
                if self.request_type != 'validate_only':
                    if self.backup:
                        self.backup_file = module.backup_local(self.path)
                    write_file(module, to_bytes(self.cert_details.get('endEntityCert')))
                    if self.full_chain_path and self.cert_details.get('chainCerts'):
                        if self.backup:
                            self.backup_full_chain_file = module.backup_local(self.full_chain_path)
                        chain_string = '\n'.join(self.cert_details.get('chainCerts')) + '\n'
                        write_file(module, to_bytes(chain_string), path=self.full_chain_path)
                    self.changed = True
        elif not os.path.exists(self.path) and self.tracking_id:
            if not module.check_mode:
                write_file(module, to_bytes(self.cert_details.get('endEntityCert')))
                if self.full_chain_path and self.cert_details.get('chainCerts'):
                    chain_string = '\n'.join(self.cert_details.get('chainCerts')) + '\n'
                    write_file(module, to_bytes(chain_string), path=self.full_chain_path)
            self.changed = True

    def dump(self):
        result = {'changed': self.changed, 'filename': self.path, 'tracking_id': self.tracking_id, 'cert_status': self.cert_status, 'serial_number': self.serial_number, 'cert_days': self.cert_days, 'cert_details': self.cert_details}
        if self.backup_file:
            result['backup_file'] = self.backup_file
            result['backup_full_chain_file'] = self.backup_full_chain_file
        return result