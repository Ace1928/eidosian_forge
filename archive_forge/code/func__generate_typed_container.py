import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _generate_typed_container(self, response):
    resp_type = response.get('type', '').lower()
    container_type = self._container_map.get(resp_type)
    if not container_type:
        raise TypeError('Unknown container type "{0}".'.format(resp_type))
    name = response.get('name')
    consumers = response.get('consumers', [])
    container_ref = response.get('container_ref')
    created = response.get('created')
    updated = response.get('updated')
    status = response.get('status')
    secret_refs = self._translate_secret_refs_from_json(response.get('secret_refs'))
    if container_type is RSAContainer:
        public_key_ref = secret_refs.get('public_key')
        private_key_ref = secret_refs.get('private_key')
        private_key_pass_ref = secret_refs.get('private_key_passphrase')
        return RSAContainer(api=self._api, name=name, consumers=consumers, container_ref=container_ref, created=created, updated=updated, status=status, public_key_ref=public_key_ref, private_key_ref=private_key_ref, private_key_passphrase_ref=private_key_pass_ref)
    elif container_type is CertificateContainer:
        certificate_ref = secret_refs.get('certificate')
        intermediates_ref = secret_refs.get('intermediates')
        private_key_ref = secret_refs.get('private_key')
        private_key_pass_ref = secret_refs.get('private_key_passphrase')
        return CertificateContainer(api=self._api, name=name, consumers=consumers, container_ref=container_ref, created=created, updated=updated, status=status, certificate_ref=certificate_ref, intermediates_ref=intermediates_ref, private_key_ref=private_key_ref, private_key_passphrase_ref=private_key_pass_ref)
    return container_type(api=self._api, name=name, secret_refs=secret_refs, consumers=consumers, container_ref=container_ref, created=created, updated=updated, status=status)