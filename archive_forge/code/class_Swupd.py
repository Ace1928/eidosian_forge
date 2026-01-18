from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
class Swupd(object):
    FILES_NOT_MATCH = 'files did not match'
    FILES_REPLACED = 'missing files were replaced'
    FILES_FIXED = 'files were fixed'
    FILES_DELETED = 'files were deleted'

    def __init__(self, module):
        self.module = module
        self.swupd_cmd = module.get_bin_path('swupd', False)
        if not self.swupd_cmd:
            module.fail_json(msg='Could not find swupd.')
        for key in module.params.keys():
            setattr(self, key, module.params[key])
        self.changed = False
        self.failed = False
        self.msg = None
        self.rc = None
        self.stderr = ''
        self.stdout = ''

    def _run_cmd(self, cmd):
        self.rc, self.stdout, self.stderr = self.module.run_command(cmd, check_rc=False)

    def _get_cmd(self, command):
        cmd = '%s %s' % (self.swupd_cmd, command)
        if self.format:
            cmd += ' --format=%s' % self.format
        if self.manifest:
            cmd += ' --manifest=%s' % self.manifest
        if self.url:
            cmd += ' --url=%s' % self.url
        else:
            if self.contenturl and command != 'check-update':
                cmd += ' --contenturl=%s' % self.contenturl
            if self.versionurl:
                cmd += ' --versionurl=%s' % self.versionurl
        return cmd

    def _is_bundle_installed(self, bundle):
        try:
            os.stat('/usr/share/clear/bundles/%s' % bundle)
        except OSError:
            return False
        return True

    def _needs_update(self):
        cmd = self._get_cmd('check-update')
        self._run_cmd(cmd)
        if self.rc == 0:
            return True
        if self.rc == 1:
            return False
        self.failed = True
        self.msg = 'Failed to check for updates'

    def _needs_verify(self):
        cmd = self._get_cmd('verify')
        self._run_cmd(cmd)
        if self.rc != 0:
            self.failed = True
            self.msg = 'Failed to check for filesystem inconsistencies.'
        if self.FILES_NOT_MATCH in self.stdout:
            return True
        return False

    def install_bundle(self, bundle):
        """Installs a bundle with `swupd bundle-add bundle`"""
        if self.module.check_mode:
            self.module.exit_json(changed=not self._is_bundle_installed(bundle))
        if self._is_bundle_installed(bundle):
            self.msg = 'Bundle %s is already installed' % bundle
            return
        cmd = self._get_cmd('bundle-add %s' % bundle)
        self._run_cmd(cmd)
        if self.rc == 0:
            self.changed = True
            self.msg = 'Bundle %s installed' % bundle
            return
        self.failed = True
        self.msg = 'Failed to install bundle %s' % bundle

    def remove_bundle(self, bundle):
        """Removes a bundle with `swupd bundle-remove bundle`"""
        if self.module.check_mode:
            self.module.exit_json(changed=self._is_bundle_installed(bundle))
        if not self._is_bundle_installed(bundle):
            self.msg = 'Bundle %s not installed'
            return
        cmd = self._get_cmd('bundle-remove %s' % bundle)
        self._run_cmd(cmd)
        if self.rc == 0:
            self.changed = True
            self.msg = 'Bundle %s removed' % bundle
            return
        self.failed = True
        self.msg = 'Failed to remove bundle %s' % bundle

    def update_os(self):
        """Updates the os with `swupd update`"""
        if self.module.check_mode:
            self.module.exit_json(changed=self._needs_update())
        if not self._needs_update():
            self.msg = 'There are no updates available'
            return
        cmd = self._get_cmd('update')
        self._run_cmd(cmd)
        if self.rc == 0:
            self.changed = True
            self.msg = 'Update successful'
            return
        self.failed = True
        self.msg = 'Failed to check for updates'

    def verify_os(self):
        """Verifies filesystem against specified or current version"""
        if self.module.check_mode:
            self.module.exit_json(changed=self._needs_verify())
        if not self._needs_verify():
            self.msg = 'No files where changed'
            return
        cmd = self._get_cmd('verify --fix')
        self._run_cmd(cmd)
        if self.rc == 0 and (self.FILES_REPLACED in self.stdout or self.FILES_FIXED in self.stdout or self.FILES_DELETED in self.stdout):
            self.changed = True
            self.msg = 'Fix successful'
            return
        self.failed = True
        self.msg = 'Failed to verify the OS'