from __future__ import absolute_import, division, print_function
import datetime
import time
import os
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import ECSClient, RestOperationException, SessionConfigurationException
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
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