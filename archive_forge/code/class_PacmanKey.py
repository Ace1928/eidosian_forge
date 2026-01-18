from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
class PacmanKey(object):

    def __init__(self, module):
        self.module = module
        self.gpg = module.get_bin_path('gpg', required=True)
        self.pacman_key = module.get_bin_path('pacman-key', required=True)
        keyid = module.params['id']
        url = module.params['url']
        data = module.params['data']
        file = module.params['file']
        keyserver = module.params['keyserver']
        verify = module.params['verify']
        force_update = module.params['force_update']
        keyring = module.params['keyring']
        state = module.params['state']
        self.keylength = 40
        keyid = self.sanitise_keyid(keyid)
        key_present = self.key_in_keyring(keyring, keyid)
        if module.check_mode:
            if state == 'present':
                changed = key_present and force_update or not key_present
                module.exit_json(changed=changed)
            elif state == 'absent':
                if key_present:
                    module.exit_json(changed=True)
                module.exit_json(changed=False)
        if state == 'present':
            if key_present and (not force_update):
                module.exit_json(changed=False)
            if data:
                file = self.save_key(data)
                self.add_key(keyring, file, keyid, verify)
                module.exit_json(changed=True)
            elif file:
                self.add_key(keyring, file, keyid, verify)
                module.exit_json(changed=True)
            elif url:
                data = self.fetch_key(url)
                file = self.save_key(data)
                self.add_key(keyring, file, keyid, verify)
                module.exit_json(changed=True)
            elif keyserver:
                self.recv_key(keyring, keyid, keyserver)
                module.exit_json(changed=True)
        elif state == 'absent':
            if key_present:
                self.remove_key(keyring, keyid)
                module.exit_json(changed=True)
            module.exit_json(changed=False)

    def is_hexadecimal(self, string):
        """Check if a given string is valid hexadecimal"""
        try:
            int(string, 16)
        except ValueError:
            return False
        return True

    def sanitise_keyid(self, keyid):
        """Sanitise given key ID.

        Strips whitespace, uppercases all characters, and strips leading `0X`.
        """
        sanitised_keyid = keyid.strip().upper().replace(' ', '').replace('0X', '')
        if len(sanitised_keyid) != self.keylength:
            self.module.fail_json(msg='key ID is not full-length: %s' % sanitised_keyid)
        if not self.is_hexadecimal(sanitised_keyid):
            self.module.fail_json(msg='key ID is not hexadecimal: %s' % sanitised_keyid)
        return sanitised_keyid

    def fetch_key(self, url):
        """Downloads a key from url"""
        response, info = fetch_url(self.module, url)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch key at %s, error was %s' % (url, info['msg']))
        return to_native(response.read())

    def recv_key(self, keyring, keyid, keyserver):
        """Receives key via keyserver"""
        cmd = [self.pacman_key, '--gpgdir', keyring, '--keyserver', keyserver, '--recv-keys', keyid]
        self.module.run_command(cmd, check_rc=True)
        self.lsign_key(keyring, keyid)

    def lsign_key(self, keyring, keyid):
        """Locally sign key"""
        cmd = [self.pacman_key, '--gpgdir', keyring]
        self.module.run_command(cmd + ['--lsign-key', keyid], check_rc=True)

    def save_key(self, data):
        """Saves key data to a temporary file"""
        tmpfd, tmpname = tempfile.mkstemp()
        self.module.add_cleanup_file(tmpname)
        tmpfile = os.fdopen(tmpfd, 'w')
        tmpfile.write(data)
        tmpfile.close()
        return tmpname

    def add_key(self, keyring, keyfile, keyid, verify):
        """Add key to pacman's keyring"""
        if verify:
            self.verify_keyfile(keyfile, keyid)
        cmd = [self.pacman_key, '--gpgdir', keyring, '--add', keyfile]
        self.module.run_command(cmd, check_rc=True)
        self.lsign_key(keyring, keyid)

    def remove_key(self, keyring, keyid):
        """Remove key from pacman's keyring"""
        cmd = [self.pacman_key, '--gpgdir', keyring, '--delete', keyid]
        self.module.run_command(cmd, check_rc=True)

    def verify_keyfile(self, keyfile, keyid):
        """Verify that keyfile matches the specified key ID"""
        if keyfile is None:
            self.module.fail_json(msg='expected a key, got none')
        elif keyid is None:
            self.module.fail_json(msg='expected a key ID, got none')
        rc, stdout, stderr = self.module.run_command([self.gpg, '--with-colons', '--with-fingerprint', '--batch', '--no-tty', '--show-keys', keyfile], check_rc=True)
        extracted_keyid = None
        for line in stdout.splitlines():
            if line.startswith('fpr:'):
                extracted_keyid = line.split(':')[9]
                break
        if extracted_keyid != keyid:
            self.module.fail_json(msg='key ID does not match. expected %s, got %s' % (keyid, extracted_keyid))

    def key_in_keyring(self, keyring, keyid):
        """Check if the key ID is in pacman's keyring"""
        rc, stdout, stderr = self.module.run_command([self.gpg, '--with-colons', '--batch', '--no-tty', '--no-default-keyring', '--keyring=%s/pubring.gpg' % keyring, '--list-keys', keyid], check_rc=False)
        if rc != 0:
            if stderr.find('No public key') >= 0:
                return False
            else:
                self.module.fail_json(msg='gpg returned an error: %s' % stderr)
        return True