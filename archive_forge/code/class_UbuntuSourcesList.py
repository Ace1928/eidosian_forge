from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
class UbuntuSourcesList(SourcesList):
    LP_API = 'https://launchpad.net/api/1.0/~%s/+archive/%s'

    def __init__(self, module):
        self.module = module
        self.codename = module.params['codename'] or distro.codename
        super(UbuntuSourcesList, self).__init__(module)
        self.apt_key_bin = self.module.get_bin_path('apt-key', required=False)
        self.gpg_bin = self.module.get_bin_path('gpg', required=False)
        if not self.apt_key_bin and (not self.gpg_bin):
            self.module.fail_json(msg='Either apt-key or gpg binary is required, but neither could be found')

    def __deepcopy__(self, memo=None):
        return UbuntuSourcesList(self.module)

    def _get_ppa_info(self, owner_name, ppa_name):
        lp_api = self.LP_API % (owner_name, ppa_name)
        headers = dict(Accept='application/json')
        response, info = fetch_url(self.module, lp_api, headers=headers)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch PPA information, error was: %s' % info['msg'])
        return json.loads(to_native(response.read()))

    def _expand_ppa(self, path):
        ppa = path.split(':')[1]
        ppa_owner = ppa.split('/')[0]
        try:
            ppa_name = ppa.split('/')[1]
        except IndexError:
            ppa_name = 'ppa'
        line = 'deb http://ppa.launchpad.net/%s/%s/ubuntu %s main' % (ppa_owner, ppa_name, self.codename)
        return (line, ppa_owner, ppa_name)

    def _key_already_exists(self, key_fingerprint):
        if self.apt_key_bin:
            locale = get_best_parsable_locale(self.module)
            APT_ENV = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LC_CTYPE=locale)
            self.module.run_command_environ_update = APT_ENV
            rc, out, err = self.module.run_command([self.apt_key_bin, 'export', key_fingerprint], check_rc=True)
            found = bool(not err or 'nothing exported' not in err)
        else:
            found = self._gpg_key_exists(key_fingerprint)
        return found

    def _gpg_key_exists(self, key_fingerprint):
        found = False
        keyfiles = ['/etc/apt/trusted.gpg']
        for other_dir in APT_KEY_DIRS:
            keyfiles.extend([os.path.join(other_dir, x) for x in os.listdir(other_dir) if not x.startswith('.')])
        for key_file in keyfiles:
            if os.path.exists(key_file):
                try:
                    rc, out, err = self.module.run_command([self.gpg_bin, '--list-packets', key_file])
                except (IOError, OSError) as e:
                    self.debug('Could check key against file %s: %s' % (key_file, to_native(e)))
                    continue
                if key_fingerprint in out:
                    found = True
                    break
        return found

    def add_source(self, line, comment='', file=None):
        if line.startswith('ppa:'):
            source, ppa_owner, ppa_name = self._expand_ppa(line)
            if source in self.repos_urls:
                return
            info = self._get_ppa_info(ppa_owner, ppa_name)
            if not self._key_already_exists(info['signing_key_fingerprint']):
                keyfile = ''
                if not self.module.check_mode:
                    if self.apt_key_bin:
                        command = [self.apt_key_bin, 'adv', '--recv-keys', '--no-tty', '--keyserver', 'hkp://keyserver.ubuntu.com:80', info['signing_key_fingerprint']]
                    else:
                        for keydir in APT_KEY_DIRS:
                            if os.path.exists(keydir):
                                break
                        else:
                            self.module.fail_json('Unable to find any existing apt gpgp repo directories, tried the following: %s' % ', '.join(APT_KEY_DIRS))
                        keyfile = '%s/%s-%s-%s.gpg' % (keydir, os.path.basename(source).replace(' ', '-'), ppa_owner, ppa_name)
                        command = [self.gpg_bin, '--no-tty', '--keyserver', 'hkp://keyserver.ubuntu.com:80', '--export', info['signing_key_fingerprint']]
                    rc, stdout, stderr = self.module.run_command(command, check_rc=True, encoding=None)
                    if keyfile:
                        if len(stdout) == 0:
                            self.module.fail_json(msg='Unable to get required signing key', rc=rc, stderr=stderr, command=command)
                        try:
                            with open(keyfile, 'wb') as f:
                                f.write(stdout)
                            self.module.log('Added repo key "%s" for apt to file "%s"' % (info['signing_key_fingerprint'], keyfile))
                        except (OSError, IOError) as e:
                            self.module.fail_json(msg='Unable to add required signing key for%s ', rc=rc, stderr=stderr, error=to_native(e))
            file = file or self._suggest_filename('%s_%s' % (line, self.codename))
        else:
            source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
            file = file or self._suggest_filename(source)
        self._add_valid_source(source, comment, file)

    def remove_source(self, line):
        if line.startswith('ppa:'):
            source = self._expand_ppa(line)[0]
        else:
            source = self._parse(line, raise_if_invalid_or_disabled=True)[2]
        self._remove_valid_source(source)

    @property
    def repos_urls(self):
        _repositories = []
        for parsed_repos in self.files.values():
            for parsed_repo in parsed_repos:
                valid = parsed_repo[1]
                enabled = parsed_repo[2]
                source_line = parsed_repo[3]
                if not valid or not enabled:
                    continue
                if source_line.startswith('ppa:'):
                    source, ppa_owner, ppa_name = self._expand_ppa(source_line)
                    _repositories.append(source)
                else:
                    _repositories.append(source_line)
        return _repositories