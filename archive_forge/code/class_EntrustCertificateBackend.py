from __future__ import absolute_import, division, print_function
import datetime
import time
import os
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import ECSClient, RestOperationException, SessionConfigurationException
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class EntrustCertificateBackend(CertificateBackend):

    def __init__(self, module, backend):
        super(EntrustCertificateBackend, self).__init__(module, backend)
        self.trackingId = None
        self.notAfter = get_relative_time_option(module.params['entrust_not_after'], 'entrust_not_after', backend=self.backend)
        if self.csr_content is None and self.csr_path is None:
            raise CertificateError('csr_path or csr_content is required for entrust provider')
        if self.csr_content is None and (not os.path.exists(self.csr_path)):
            raise CertificateError('The certificate signing request file {0} does not exist'.format(self.csr_path))
        self._ensure_csr_loaded()
        self.csr_org = None
        if self.backend == 'cryptography':
            csr_subject_orgs = self.csr.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
            if len(csr_subject_orgs) == 1:
                self.csr_org = csr_subject_orgs[0].value
            elif len(csr_subject_orgs) > 1:
                self.module.fail_json(msg="Entrust provider does not currently support multiple validated organizations. Multiple organizations found in Subject DN: '{0}'. ".format(self.csr.subject))
        if self.csr_org is None:
            self.csr_org = ''
        try:
            self.ecs_client = ECSClient(entrust_api_user=self.module.params['entrust_api_user'], entrust_api_key=self.module.params['entrust_api_key'], entrust_api_cert=self.module.params['entrust_api_client_cert_path'], entrust_api_cert_key=self.module.params['entrust_api_client_cert_key_path'], entrust_api_specification_path=self.module.params['entrust_api_specification_path'])
        except SessionConfigurationException as e:
            module.fail_json(msg='Failed to initialize Entrust Provider: {0}'.format(to_native(e.message)))

    def generate_certificate(self):
        """(Re-)Generate certificate."""
        body = {}
        if self.csr_content is not None:
            body['csr'] = to_native(self.csr_content)
        else:
            with open(self.csr_path, 'r') as csr_file:
                body['csr'] = csr_file.read()
        body['certType'] = self.module.params['entrust_cert_type']
        expiry = self.notAfter
        if not expiry:
            gmt_now = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
            expiry = gmt_now + datetime.timedelta(days=365)
        expiry_iso3339 = expiry.strftime('%Y-%m-%dT%H:%M:%S.00Z')
        body['certExpiryDate'] = expiry_iso3339
        body['org'] = self.csr_org
        body['tracking'] = {'requesterName': self.module.params['entrust_requester_name'], 'requesterEmail': self.module.params['entrust_requester_email'], 'requesterPhone': self.module.params['entrust_requester_phone']}
        try:
            result = self.ecs_client.NewCertRequest(Body=body)
            self.trackingId = result.get('trackingId')
        except RestOperationException as e:
            self.module.fail_json(msg='Failed to request new certificate from Entrust Certificate Services (ECS): {0}'.format(to_native(e.message)))
        self.cert_bytes = to_bytes(result.get('endEntityCert'))
        self.cert = load_certificate(path=None, content=self.cert_bytes, backend=self.backend)

    def get_certificate_data(self):
        """Return bytes for self.cert."""
        return self.cert_bytes

    def needs_regeneration(self):
        parent_check = super(EntrustCertificateBackend, self).needs_regeneration()
        try:
            cert_details = self._get_cert_details()
        except RestOperationException as e:
            self.module.fail_json(msg='Failed to get status of existing certificate from Entrust Certificate Services (ECS): {0}.'.format(to_native(e.message)))
        status = cert_details.get('status', False)
        if status == 'EXPIRED' or status == 'SUSPENDED' or status == 'REVOKED':
            return True
        if self.module.params['entrust_cert_type'] and cert_details.get('certType') and (self.module.params['entrust_cert_type'] != cert_details.get('certType')):
            return True
        return parent_check

    def _get_cert_details(self):
        cert_details = {}
        try:
            self._ensure_existing_certificate_loaded()
        except Exception as dummy:
            return
        if self.existing_certificate:
            serial_number = None
            expiry = None
            if self.backend == 'cryptography':
                serial_number = '{0:X}'.format(cryptography_serial_number_of_cert(self.existing_certificate))
                expiry = self.existing_certificate.not_valid_after
            expiry_iso3339 = expiry.strftime('%Y-%m-%dT%H:%M:%S.00Z')
            cert_details['expiresAfter'] = expiry_iso3339
            if self.trackingId is None and serial_number is not None:
                cert_results = self.ecs_client.GetCertificates(serialNumber=serial_number).get('certificates', {})
                if len(cert_results) == 1:
                    self.trackingId = cert_results[0].get('trackingId')
        if self.trackingId is not None:
            cert_details.update(self.ecs_client.GetCertificate(trackingId=self.trackingId))
        return cert_details