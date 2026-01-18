from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class RootUserManager(BaseManager):

    def exec_module(self):
        if not HAS_CRYPTO:
            raise F5ModuleError("An installed and up-to-date python 'cryptography' package is required to change the 'root' password.")
        start = datetime.now().isoformat()
        version = tmos_version(self.client)
        changed = False
        result = dict()
        state = self.want.state
        if state == 'present':
            changed = self.present()
        elif state == 'absent':
            raise F5ModuleError('You may not remove the root user.')
        reportable = ReportableChanges(params=self.changes.to_return())
        changes = reportable.to_return()
        result.update(**changes)
        result.update(dict(changed=changed))
        self._announce_deprecations(result)
        send_teem(start, self.client, self.module, version)
        return result

    def exists(self):
        return True

    def update(self):
        public_key = self.get_public_key_from_device()
        public_key = self.extract_key(public_key)
        encrypted = self.encrypt_password_change_file(public_key, self.want.password_credential)
        self.upload_to_device(encrypted, self.want.temp_upload_file)
        result = self.update_on_device()
        self.remove_uploaded_file_from_device(self.want.temp_upload_file)
        return result

    def encrypt_password_change_file(self, public_key, password):
        pub = serialization.load_pem_public_key(to_bytes(public_key), backend=default_backend())
        message = to_bytes('{0}\n{0}\n'.format(password))
        ciphertext = pub.encrypt(message, padding.PKCS1v15())
        return BytesIO(ciphertext)

    def extract_key(self, content):
        """Extracts the public key from the openssl command output over REST

        The REST output includes some extra output that is not relevant to the
        public key. This function attempts to only return the valid public key
        data from the openssl output

        Args:
            content: The output from the REST API command to view the public key.

        Returns:
            string: The discovered public key
        """
        lines = content.split('\n')
        start = lines.index('-----BEGIN PUBLIC KEY-----')
        end = lines.index('-----END PUBLIC KEY-----')
        result = '\n'.join(lines[start:end + 1])
        return result

    def update_on_device(self):
        errors = ['Bad password', 'password change canceled', 'based on a dictionary word']
        openssl = ['openssl', 'pkeyutl', '-in', '/var/config/rest/downloads/{0}'.format(self.want.temp_upload_file), '-decrypt', '-inkey', '/config/ssl/ssl.key/default.key']
        cmd = '-c "{0} | tmsh modify auth password root"'.format(' '.join(openssl))
        params = dict(command='run', utilCmdArgs=cmd)
        uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
            if 'commandResult' in response:
                if any((x for x in errors if x in response['commandResult'])):
                    raise F5ModuleError(response['commandResult'])
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        return True

    def upload_to_device(self, content, name):
        """Uploads a file-like object via the REST API to a given filename

        Args:
            content: The file-like object whose content to upload
            name: The remote name of the file to store the content in. The
                  final location of the file will be in /var/config/rest/downloads.

        Returns:
            void
        """
        url = 'https://{0}:{1}/mgmt/shared/file-transfer/uploads'.format(self.client.provider['server'], self.client.provider['server_port'])
        try:
            upload_file(self.client, url, content, name)
        except F5ModuleError:
            raise F5ModuleError('Failed to upload the file.')

    def remove_uploaded_file_from_device(self, name):
        filepath = '/var/config/rest/downloads/{0}'.format(name)
        params = {'command': 'run', 'utilCmdArgs': filepath}
        uri = 'https://{0}:{1}/mgmt/tm/util/unix-rm'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def get_public_key_from_device(self):
        cmd = '-c "openssl rsa -in /config/ssl/ssl.key/default.key -pubout"'
        params = dict(command='run', utilCmdArgs=cmd)
        uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        if 'commandResult' in response:
            return response['commandResult']
        return None