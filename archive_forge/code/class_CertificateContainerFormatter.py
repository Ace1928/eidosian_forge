import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
class CertificateContainerFormatter(formatter.EntityFormatter):
    _get_generic_data = ContainerFormatter._get_formatted_data

    def _get_generic_columns(self):
        return ContainerFormatter.columns
    columns = ('Container href', 'Name', 'Created', 'Status', 'Type', 'Certificate', 'Intermediates', 'Private Key', 'PK Passphrase', 'Consumers')

    def _get_formatted_data(self):
        formatted_certificate = None
        formatted_private_key = None
        formatted_pkp = None
        formatted_intermediates = None
        formatted_consumers = None
        if self.certificate:
            formatted_certificate = self.certificate.secret_ref
        if self.intermediates:
            formatted_intermediates = self.intermediates.secret_ref
        if self.private_key:
            formatted_private_key = self.private_key.secret_ref
        if self.private_key_passphrase:
            formatted_pkp = self.private_key_passphrase.secret_ref
        if self.consumers:
            formatted_consumers = '\n'.join((str(c) for c in self.consumers))
        data = (self.container_ref, self.name, self.created, self.status, self._type, formatted_certificate, formatted_intermediates, formatted_private_key, formatted_pkp, formatted_consumers)
        return data