from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.common.text.converters import to_native
class YumRepo(object):
    module = None
    params = None
    section = None
    repofile = configparser.RawConfigParser()
    allowed_params = ['async', 'bandwidth', 'baseurl', 'cost', 'deltarpm_metadata_percentage', 'deltarpm_percentage', 'enabled', 'enablegroups', 'exclude', 'failovermethod', 'gpgcakey', 'gpgcheck', 'gpgkey', 'module_hotfixes', 'http_caching', 'include', 'includepkgs', 'ip_resolve', 'keepalive', 'keepcache', 'metadata_expire', 'metadata_expire_filter', 'metalink', 'mirrorlist', 'mirrorlist_expire', 'name', 'password', 'priority', 'protect', 'proxy', 'proxy_password', 'proxy_username', 'repo_gpgcheck', 'retries', 's3_enabled', 'skip_if_unavailable', 'sslcacert', 'ssl_check_cert_permissions', 'sslclientcert', 'sslclientkey', 'sslverify', 'throttle', 'timeout', 'ui_repoid_vars', 'username']
    list_params = ['exclude', 'includepkgs']

    def __init__(self, module):
        self.module = module
        self.params = self.module.params
        self.section = self.params['repoid']
        repos_dir = self.params['reposdir']
        if not os.path.isdir(repos_dir):
            self.module.fail_json(msg="Repo directory '%s' does not exist." % repos_dir)
        self.params['dest'] = os.path.join(repos_dir, '%s.repo' % self.params['file'])
        if os.path.isfile(self.params['dest']):
            self.repofile.read(self.params['dest'])

    def add(self):
        if self.repofile.has_section(self.section):
            self.repofile.remove_section(self.section)
        self.repofile.add_section(self.section)
        req_params = (self.params['baseurl'], self.params['metalink'], self.params['mirrorlist'])
        if req_params == (None, None, None):
            self.module.fail_json(msg="Parameter 'baseurl', 'metalink' or 'mirrorlist' is required for adding a new repo.")
        for key, value in sorted(self.params.items()):
            if key in self.list_params and isinstance(value, list):
                value = ' '.join(value)
            elif isinstance(value, bool):
                value = int(value)
            if value is not None and key in self.allowed_params:
                if key == 'keepcache':
                    self.module.deprecate("'keepcache' parameter is deprecated.", version='2.20')
                self.repofile.set(self.section, key, value)

    def save(self):
        if len(self.repofile.sections()):
            try:
                with open(self.params['dest'], 'w') as fd:
                    self.repofile.write(fd)
            except IOError as e:
                self.module.fail_json(msg='Problems handling file %s.' % self.params['dest'], details=to_native(e))
        else:
            try:
                os.remove(self.params['dest'])
            except OSError as e:
                self.module.fail_json(msg='Cannot remove empty repo file %s.' % self.params['dest'], details=to_native(e))

    def remove(self):
        if self.repofile.has_section(self.section):
            self.repofile.remove_section(self.section)

    def dump(self):
        repo_string = ''
        for section in sorted(self.repofile.sections()):
            repo_string += '[%s]\n' % section
            for key, value in sorted(self.repofile.items(section)):
                repo_string += '%s = %s\n' % (key, value)
            repo_string += '\n'
        return repo_string