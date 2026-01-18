from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
class NetAppESeriesServerCertificate(NetAppESeriesModule):
    RESET_SSL_CONFIG_TIMEOUT_SEC = 3 * 60

    def __init__(self):
        ansible_options = dict(controller=dict(type='str', required=False, choices=['A', 'B']), certificates=dict(type='list', required=False), passphrase=dict(type='str', required=False, no_log=True))
        super(NetAppESeriesServerCertificate, self).__init__(ansible_options=ansible_options, web_services_version='05.00.0000.0000', supports_check_mode=True)
        args = self.module.params
        self.controller = args['controller']
        self.certificates = args['certificates'] if 'certificates' in args.keys() else list()
        self.passphrase = args['passphrase'] if 'passphrase' in args.keys() else None
        self.url_path_prefix = ''
        self.url_path_suffix = ''
        if self.is_proxy():
            if self.ssid.lower() in ['0', 'proxy']:
                self.url_path_suffix = '?controller=auto'
            elif self.controller is not None:
                self.url_path_prefix = 'storage-systems/%s/forward/devmgr/v2/' % self.ssid
                self.url_path_suffix = '?controller=%s' % self.controller.lower()
            else:
                self.module.fail_json(msg="Invalid options! You must specify which controller's certificates to modify. Array [%s]." % self.ssid)
        elif self.controller is None:
            self.module.fail_json(msg="Invalid options! You must specify which controller's certificates to modify. Array [%s]." % self.ssid)
        self.cache_get_current_certificates = None
        self.cache_is_controller_alternate = None
        self.cache_is_public_server_certificate_signed = None

    def get_controllers(self):
        """Retrieve a mapping of controller labels to their controller slot."""
        controllers_dict = {}
        controllers = []
        try:
            rc, controllers = self.request('storage-systems/%s/controllers' % self.ssid)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve the controller settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
        for controller in controllers:
            slot = controller['physicalLocation']['slot']
            letter = chr(slot + 64)
            controllers_dict.update({letter: slot})
        return controllers_dict

    def check_controller(self):
        """Is the effected controller the alternate controller."""
        controllers_info = self.get_controllers()
        try:
            rc, about = self.request('utils/about', rest_api_path=self.DEFAULT_BASE_PATH)
            self.url_path_suffix = '?alternate=%s' % ('true' if controllers_info[self.controller] != about['controllerPosition'] else 'false')
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve accessing controller slot information. Array [%s].' % self.ssid)

    @staticmethod
    def sanitize_distinguished_name(dn):
        """Generate a sorted distinguished name string to account for different formats/orders."""
        dn = re.sub(' *= *', '=', dn).lower()
        dn = re.sub(', *(?=[a-zA-Z]+={1})', '---SPLIT_MARK---', dn)
        dn_parts = dn.split('---SPLIT_MARK---')
        dn_parts.sort()
        return ','.join(dn_parts)

    def certificate_info_from_file(self, path):
        """Determine the certificate info from the provided filepath."""
        certificates_info = {}
        try:
            with open(path, 'r') as fh:
                line = fh.readline()
                while line != '':
                    if re.search('^-+BEGIN CERTIFICATE-+$', line):
                        certificate = line
                        line = fh.readline()
                        while not re.search('^-+END CERTIFICATE-+$', line):
                            if line == '':
                                self.module.fail_json(msg='Invalid certificate! Path [%s]. Array [%s].' % (path, self.ssid))
                            certificate += line
                            line = fh.readline()
                        certificate += line
                        if not six.PY2:
                            certificate = six.b(certificate)
                        info = x509.load_pem_x509_certificate(certificate, default_backend())
                        certificates_info.update(self.certificate_info(info, certificate, path))
                    elif re.search('^-+BEGIN.*PRIVATE KEY-+$', line):
                        pkcs8 = 'BEGIN PRIVATE KEY' in line
                        pkcs8_encrypted = 'BEGIN ENCRYPTED PRIVATE KEY' in line
                        key = line
                        line = fh.readline()
                        while not re.search('^-+END.*PRIVATE KEY-+$', line):
                            if line == '':
                                self.module.fail_json(msg='Invalid certificate! Array [%s].' % self.ssid)
                            key += line
                            line = fh.readline()
                        key += line
                        if not six.PY2:
                            key = six.b(key)
                            if self.passphrase:
                                self.passphrase = six.b(self.passphrase)
                        if pkcs8 or pkcs8_encrypted:
                            try:
                                if pkcs8:
                                    crypto_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
                                else:
                                    crypto_key = serialization.load_pem_private_key(key, password=self.passphrase, backend=default_backend())
                            except ValueError as error:
                                self.module.fail_json(msg='Failed to load%sPKCS8 encoded private key. %s Error [%s].' % (' encrypted ' if pkcs8_encrypted else ' ', 'Check passphrase.' if pkcs8_encrypted else '', error))
                            key = crypto_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption())
                        if 'private_key' in certificates_info.keys() and certificates_info['private_key'] != key:
                            self.module.fail_json(msg='Multiple private keys have been provided! Array [%s]' % self.ssid)
                        else:
                            certificates_info.update({'private_key': key})
                    line = fh.readline()
            if len(certificates_info) == 0:
                raise Exception('Failed to discover a valid PEM encoded certificate or private key!')
        except Exception as error:
            try:
                with open(path, 'rb') as fh:
                    cert_info = x509.load_der_x509_certificate(fh.read(), default_backend())
                    cert_data = cert_info.public_bytes(serialization.Encoding.PEM)
                    certificates_info.update(self.certificate_info(cert_info, cert_data, path))
                if len(certificates_info) == 0:
                    raise Exception('Failed to discover a valid DER encoded certificate!')
            except Exception as error:
                try:
                    with open(path, 'rb') as fh:
                        crypto_key = serialization.load_der_public_key(fh.read(), backend=default_backend())
                        key = crypto_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption())
                        certificates_info.update({'private_key': key})
                except Exception as error:
                    self.module.fail_json(msg='Invalid file type! File is neither PEM or DER encoded certificate/private key. Path [%s]. Array [%s]. Error [%s].' % (path, self.ssid, to_native(error)))
        return certificates_info

    def certificate_info(self, info, data, path):
        """Load x509 certificate that is either encoded DER or PEM encoding and return the certificate fingerprint."""
        fingerprint = binascii.hexlify(info.fingerprint(info.signature_hash_algorithm)).decode('utf-8')
        return {self.sanitize_distinguished_name(info.subject.rfc4514_string()): {'alias': fingerprint, 'fingerprint': fingerprint, 'certificate': data, 'path': path, 'issuer': self.sanitize_distinguished_name(info.issuer.rfc4514_string())}}

    def get_current_certificates(self):
        """Determine the server certificates that exist on the storage system."""
        if self.cache_get_current_certificates is None:
            current_certificates = []
            try:
                rc, current_certificates = self.request(self.url_path_prefix + 'certificates/server%s' % self.url_path_suffix)
            except Exception as error:
                self.module.fail_json(msg='Failed to retrieve server certificates. Array [%s].' % self.ssid)
            self.cache_get_current_certificates = {}
            for certificate in current_certificates:
                certificate.update({'issuer': self.sanitize_distinguished_name(certificate['issuerDN'])})
                self.cache_get_current_certificates.update({self.sanitize_distinguished_name(certificate['subjectDN']): certificate})
        return self.cache_get_current_certificates

    def is_public_server_certificate_signed(self):
        """Return whether the public server certificate is signed."""
        if self.cache_is_public_server_certificate_signed is None:
            current_certificates = self.get_current_certificates()
            for certificate in current_certificates:
                if current_certificates[certificate]['alias'] == 'jetty':
                    self.cache_is_public_server_certificate_signed = current_certificates[certificate]['type'] == 'caSigned'
                    break
        return self.cache_is_public_server_certificate_signed

    def get_expected_certificates(self):
        """Determine effected certificates and return certificate list in the required submission order."""
        certificates_info = {}
        existing_certificates = self.get_current_certificates()
        private_key = None
        if self.certificates:
            for path in self.certificates:
                info = self.certificate_info_from_file(path)
                if 'private_key' in info.keys():
                    if private_key is not None and info['private_key'] != private_key:
                        self.module.fail_json(msg='Multiple private keys have been provided! Array [%s]' % self.ssid)
                    else:
                        private_key = info.pop('private_key')
                certificates_info.update(info)
        ordered_certificates_info = [dict] * len(certificates_info)
        ordered_certificates_info_index = len(certificates_info) - 1
        while certificates_info:
            for certificate_subject in certificates_info.keys():
                remaining_issuer_list = [info['issuer'] for subject, info in existing_certificates.items()]
                for subject, info in certificates_info.items():
                    remaining_issuer_list.append(info['issuer'])
                if certificate_subject not in remaining_issuer_list:
                    ordered_certificates_info[ordered_certificates_info_index] = certificates_info[certificate_subject]
                    certificates_info.pop(certificate_subject)
                    ordered_certificates_info_index -= 1
                    break
            else:
                for certificate_subject in certificates_info.keys():
                    ordered_certificates_info[ordered_certificates_info_index] = certificates_info[certificate_subject]
                    ordered_certificates_info_index -= 1
                break
        return {'private_key': private_key, 'certificates': ordered_certificates_info}

    def determine_changes(self):
        """Determine certificates that need to be added or removed from storage system's server certificates database."""
        if not self.is_proxy():
            self.check_controller()
        existing_certificates = self.get_current_certificates()
        expected = self.get_expected_certificates()
        certificates = expected['certificates']
        changes = {'change_required': False, 'signed_cert': True if certificates else False, 'private_key': expected['private_key'], 'public_cert': None, 'add_certs': [], 'remove_certs': []}
        if certificates:
            for existing_certificate_subject, existing_certificate in existing_certificates.items():
                changes['remove_certs'].append(existing_certificate['alias'])
            last_certificate_index = len(certificates) - 1
            for certificate_index, certificate in enumerate(certificates):
                for existing_certificate_subject, existing_certificate in existing_certificates.items():
                    if certificate_index == last_certificate_index:
                        if existing_certificate['alias'] == 'jetty':
                            if certificate['fingerprint'] != existing_certificate['shaFingerprint'] and certificate['fingerprint'] != existing_certificate['sha256Fingerprint']:
                                changes['change_required'] = True
                            changes['public_cert'] = certificate
                            changes['remove_certs'].remove(existing_certificate['alias'])
                            break
                    elif certificate['alias'] == existing_certificate['alias']:
                        if certificate['fingerprint'] != existing_certificate['shaFingerprint'] and certificate['fingerprint'] != existing_certificate['sha256Fingerprint']:
                            changes['add_certs'].append(certificate)
                            changes['change_required'] = True
                        changes['remove_certs'].remove(existing_certificate['alias'])
                        break
                else:
                    changes['add_certs'].append(certificate)
                    changes['change_required'] = True
        elif self.is_public_server_certificate_signed():
            changes['change_required'] = True
        return changes

    def apply_self_signed_certificate(self):
        """Install self-signed server certificate which is generated by the storage system itself."""
        try:
            rc, resp = self.request(self.url_path_prefix + 'certificates/reset%s' % self.url_path_suffix, method='POST')
        except Exception as error:
            self.module.fail_json(msg='Failed to reset SSL configuration back to a self-signed certificate! Array [%s]. Error [%s].' % (self.ssid, error))

    def apply_signed_certificate(self, public_cert, private_key):
        """Install authoritative signed server certificate whether csr is generated by storage system or not."""
        if private_key is None:
            headers, data = create_multipart_formdata([('file', 'signed_server_certificate', public_cert['certificate'])])
        else:
            headers, data = create_multipart_formdata([('file', 'signed_server_certificate', public_cert['certificate']), ('privateKey', 'private_key', private_key)])
        try:
            rc, resp = self.request(self.url_path_prefix + 'certificates/server%s&replaceMainServerCertificate=true' % self.url_path_suffix, method='POST', headers=headers, data=data)
        except Exception as error:
            self.module.fail_json(msg='Failed to upload signed server certificate! Array [%s]. Error [%s].' % (self.ssid, error))

    def upload_authoritative_certificates(self, certificate):
        """Install all authoritative certificates."""
        headers, data = create_multipart_formdata([['file', certificate['alias'], certificate['certificate']]])
        try:
            rc, resp = self.request(self.url_path_prefix + 'certificates/server%s&alias=%s' % (self.url_path_suffix, certificate['alias']), method='POST', headers=headers, data=data)
        except Exception as error:
            self.module.fail_json(msg='Failed to upload certificate authority! Array [%s]. Error [%s].' % (self.ssid, error))

    def remove_authoritative_certificates(self, alias):
        """Delete all authoritative certificates."""
        try:
            rc, resp = self.request(self.url_path_prefix + 'certificates/server/%s%s' % (alias, self.url_path_suffix), method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to delete certificate authority! Array [%s]. Error [%s].' % (self.ssid, error))

    def reload_ssl_configuration(self):
        """Asynchronously reloads the SSL configuration."""
        self.request(self.url_path_prefix + 'certificates/reload%s' % self.url_path_suffix, method='POST', ignore_errors=True)
        for retry in range(int(self.RESET_SSL_CONFIG_TIMEOUT_SEC / 3)):
            try:
                rc, current_certificates = self.request(self.url_path_prefix + 'certificates/server%s' % self.url_path_suffix)
            except Exception as error:
                sleep(3)
                continue
            break
        else:
            self.module.fail_json(msg='Failed to retrieve server certificates. Array [%s].' % self.ssid)

    def apply(self):
        """Apply state changes to the storage array's truststore."""
        if not HAS_CRYPTOGRAPHY:
            self.module.fail_json(msg='Python cryptography package are missing!')
        major, minor, patch = [int(item) for item in str(cryptography.__version__).split('.')]
        if major < 2 or (major == 2 and minor < 5):
            self.module.fail_json(msg='Python cryptography package version must greater than version 2.5! Version [%s].' % cryptography.__version__)
        changes = self.determine_changes()
        if changes['change_required'] and (not self.module.check_mode):
            if changes['signed_cert']:
                for certificate in changes['add_certs']:
                    self.upload_authoritative_certificates(certificate)
                for certificate_alias in changes['remove_certs']:
                    self.remove_authoritative_certificates(certificate_alias)
                if changes['public_cert']:
                    self.apply_signed_certificate(changes['public_cert'], changes['private_key'])
                    self.reload_ssl_configuration()
            else:
                self.apply_self_signed_certificate()
                self.reload_ssl_configuration()
        self.module.exit_json(changed=changes['change_required'], signed_server_certificate=changes['signed_cert'], added_certificates=[cert['alias'] for cert in changes['add_certs']], removed_certificates=changes['remove_certs'])