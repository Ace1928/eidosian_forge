from __future__ import absolute_import, division, print_function
import hashlib
import io
import json
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule, to_bytes
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.jenkins import download_updates_file
class JenkinsPlugin(object):

    def __init__(self, module):
        self.module = module
        self.params = self.module.params
        self.url = self.params['url']
        self.timeout = self.params['timeout']
        self.crumb = {}
        self.cookies = None
        if self._csrf_enabled():
            self.cookies = cookiejar.LWPCookieJar()
            self.crumb = self._get_crumb()
        self._get_installed_plugins()

    def _csrf_enabled(self):
        csrf_data = self._get_json_data('%s/%s' % (self.url, 'api/json'), 'CSRF')
        if 'useCrumbs' not in csrf_data:
            self.module.fail_json(msg='Required fields not found in the Crumbs response.', details=csrf_data)
        return csrf_data['useCrumbs']

    def _get_json_data(self, url, what, **kwargs):
        r = self._get_url_data(url, what, **kwargs)
        try:
            json_data = json.loads(to_native(r.read()))
        except Exception as e:
            self.module.fail_json(msg='Cannot parse %s JSON data.' % what, details=to_native(e))
        return json_data

    def _get_urls_data(self, urls, what=None, msg_status=None, msg_exception=None, **kwargs):
        if msg_status is None:
            msg_status = 'Cannot get %s' % what
        if msg_exception is None:
            msg_exception = 'Retrieval of %s failed.' % what
        errors = {}
        for url in urls:
            err_msg = None
            try:
                self.module.debug('fetching url: %s' % url)
                response, info = fetch_url(self.module, url, timeout=self.timeout, cookies=self.cookies, headers=self.crumb, **kwargs)
                if info['status'] == 200:
                    return response
                else:
                    err_msg = '%s. fetching url %s failed. response code: %s' % (msg_status, url, info['status'])
                    if info['status'] > 400:
                        err_msg = '%s. response body: %s' % (err_msg, info['body'])
            except Exception as e:
                err_msg = '%s. fetching url %s failed. error msg: %s' % (msg_status, url, to_native(e))
            finally:
                if err_msg is not None:
                    self.module.debug(err_msg)
                    errors[url] = err_msg
        self.module.fail_json(msg=msg_exception, details=errors)

    def _get_url_data(self, url, what=None, msg_status=None, msg_exception=None, dont_fail=False, **kwargs):
        if msg_status is None:
            msg_status = 'Cannot get %s' % what
        if msg_exception is None:
            msg_exception = 'Retrieval of %s failed.' % what
        try:
            response, info = fetch_url(self.module, url, timeout=self.timeout, cookies=self.cookies, headers=self.crumb, **kwargs)
            if info['status'] != 200:
                if dont_fail:
                    raise FailedInstallingWithPluginManager(info['msg'])
                else:
                    self.module.fail_json(msg=msg_status, details=info['msg'])
        except Exception as e:
            if dont_fail:
                raise FailedInstallingWithPluginManager(e)
            else:
                self.module.fail_json(msg=msg_exception, details=to_native(e))
        return response

    def _get_crumb(self):
        crumb_data = self._get_json_data('%s/%s' % (self.url, 'crumbIssuer/api/json'), 'Crumb')
        if 'crumbRequestField' in crumb_data and 'crumb' in crumb_data:
            ret = {crumb_data['crumbRequestField']: crumb_data['crumb']}
        else:
            self.module.fail_json(msg='Required fields not found in the Crum response.', details=crumb_data)
        return ret

    def _get_installed_plugins(self):
        plugins_data = self._get_json_data('%s/%s' % (self.url, 'pluginManager/api/json?depth=1'), 'list of plugins')
        if 'plugins' not in plugins_data:
            self.module.fail_json(msg='No valid plugin data found.')
        self.is_installed = False
        self.is_pinned = False
        self.is_enabled = False
        for p in plugins_data['plugins']:
            if p['shortName'] == self.params['name']:
                self.is_installed = True
                if p['pinned']:
                    self.is_pinned = True
                if p['enabled']:
                    self.is_enabled = True
                break

    def _install_with_plugin_manager(self):
        if not self.module.check_mode:
            install_script = 'd = Jenkins.instance.updateCenter.getPlugin("%s").deploy(); d.get();' % self.params['name']
            if self.params['with_dependencies']:
                install_script = 'Jenkins.instance.updateCenter.getPlugin("%s").getNeededDependencies().each{it.deploy()}; %s' % (self.params['name'], install_script)
            script_data = {'script': install_script}
            data = urlencode(script_data)
            r = self._get_url_data('%s/scriptText' % self.url, msg_status='Cannot install plugin.', msg_exception='Plugin installation has failed.', data=data, dont_fail=True)
            hpi_file = '%s/plugins/%s.hpi' % (self.params['jenkins_home'], self.params['name'])
            if os.path.isfile(hpi_file):
                os.remove(hpi_file)

    def install(self):
        changed = False
        plugin_file = '%s/plugins/%s.jpi' % (self.params['jenkins_home'], self.params['name'])
        if not self.is_installed and self.params['version'] in [None, 'latest']:
            try:
                self._install_with_plugin_manager()
                changed = True
            except FailedInstallingWithPluginManager:
                pass
        if not changed:
            if not os.path.isdir(self.params['jenkins_home']):
                self.module.fail_json(msg="Jenkins home directory doesn't exist.")
            checksum_old = None
            if os.path.isfile(plugin_file):
                with open(plugin_file, 'rb') as plugin_fh:
                    plugin_content = plugin_fh.read()
                checksum_old = hashlib.sha1(plugin_content).hexdigest()
            if self.params['version'] in [None, 'latest']:
                plugin_urls = self._get_latest_plugin_urls()
            else:
                plugin_urls = self._get_versioned_plugin_urls()
            if self.params['updates_expiration'] == 0 or self.params['version'] not in [None, 'latest'] or checksum_old is None:
                r = self._download_plugin(plugin_urls)
                if checksum_old is None:
                    if not self.module.check_mode:
                        self._write_file(plugin_file, r)
                    changed = True
                else:
                    data = r.read()
                    checksum_new = hashlib.sha1(data).hexdigest()
                    if checksum_old != checksum_new:
                        if not self.module.check_mode:
                            self._write_file(plugin_file, data)
                        changed = True
            elif self.params['version'] == 'latest':
                plugin_data = self._download_updates()
                if checksum_old != to_bytes(plugin_data['sha1']):
                    if not self.module.check_mode:
                        r = self._download_plugin(plugin_urls)
                        self._write_file(plugin_file, r)
                    changed = True
        if os.path.isfile(plugin_file):
            params = {'dest': plugin_file}
            params.update(self.params)
            file_args = self.module.load_file_common_arguments(params)
            if not self.module.check_mode:
                changed = self.module.set_fs_attributes_if_different(file_args, changed)
            else:
                changed = True
        return changed

    def _get_latest_plugin_urls(self):
        urls = []
        for base_url in self.params['updates_url']:
            for update_segment in self.params['latest_plugins_url_segments']:
                urls.append('{0}/{1}/{2}.hpi'.format(base_url, update_segment, self.params['name']))
        return urls

    def _get_versioned_plugin_urls(self):
        urls = []
        for base_url in self.params['updates_url']:
            for versioned_segment in self.params['versioned_plugins_url_segments']:
                urls.append('{0}/{1}/{2}/{3}/{2}.hpi'.format(base_url, versioned_segment, self.params['name'], self.params['version']))
        return urls

    def _get_update_center_urls(self):
        urls = []
        for base_url in self.params['updates_url']:
            for update_json in self.params['update_json_url_segment']:
                urls.append('{0}/{1}'.format(base_url, update_json))
        return urls

    def _download_updates(self):
        try:
            updates_file, download_updates = download_updates_file(self.params['updates_expiration'])
        except OSError as e:
            self.module.fail_json(msg='Cannot create temporal directory.', details=to_native(e))
        if download_updates:
            urls = self._get_update_center_urls()
            r = self._get_urls_data(urls, msg_status='Remote updates not found.', msg_exception='Updates download failed.')
            tmp_update_fd, tmp_updates_file = tempfile.mkstemp()
            os.write(tmp_update_fd, r.read())
            try:
                os.close(tmp_update_fd)
            except IOError as e:
                self.module.fail_json(msg='Cannot close the tmp updates file %s.' % tmp_updates_file, details=to_native(e))
        else:
            tmp_updates_file = updates_file
        try:
            f = io.open(tmp_updates_file, encoding='utf-8')
            dummy = f.readline()
            data = json.loads(f.readline())
        except IOError as e:
            self.module.fail_json(msg='Cannot open%s updates file.' % (' temporary' if tmp_updates_file != updates_file else ''), details=to_native(e))
        except Exception as e:
            self.module.fail_json(msg='Cannot load JSON data from the%s updates file.' % (' temporary' if tmp_updates_file != updates_file else ''), details=to_native(e))
        if tmp_updates_file != updates_file:
            self.module.atomic_move(tmp_updates_file, updates_file)
        if not data.get('plugins', {}).get(self.params['name']):
            self.module.fail_json(msg='Cannot find plugin data in the updates file.')
        return data['plugins'][self.params['name']]

    def _download_plugin(self, plugin_urls):
        return self._get_urls_data(plugin_urls, msg_status='Plugin not found.', msg_exception='Plugin download failed.')

    def _write_file(self, f, data):
        tmp_f_fd, tmp_f = tempfile.mkstemp()
        if isinstance(data, (text_type, binary_type)):
            os.write(tmp_f_fd, data)
        else:
            os.write(tmp_f_fd, data.read())
        try:
            os.close(tmp_f_fd)
        except IOError as e:
            self.module.fail_json(msg='Cannot close the temporal plugin file %s.' % tmp_f, details=to_native(e))
        self.module.atomic_move(tmp_f, f)

    def uninstall(self):
        changed = False
        if self.is_installed:
            if not self.module.check_mode:
                self._pm_query('doUninstall', 'Uninstallation')
            changed = True
        return changed

    def pin(self):
        return self._pinning('pin')

    def unpin(self):
        return self._pinning('unpin')

    def _pinning(self, action):
        changed = False
        if action == 'pin' and (not self.is_pinned) or (action == 'unpin' and self.is_pinned):
            if not self.module.check_mode:
                self._pm_query(action, '%sning' % action.capitalize())
            changed = True
        return changed

    def enable(self):
        return self._enabling('enable')

    def disable(self):
        return self._enabling('disable')

    def _enabling(self, action):
        changed = False
        if action == 'enable' and (not self.is_enabled) or (action == 'disable' and self.is_enabled):
            if not self.module.check_mode:
                self._pm_query('make%sd' % action.capitalize(), '%sing' % action[:-1].capitalize())
            changed = True
        return changed

    def _pm_query(self, action, msg):
        url = '%s/pluginManager/plugin/%s/%s' % (self.params['url'], self.params['name'], action)
        self._get_url_data(url, msg_status='Plugin not found. %s' % url, msg_exception='%s has failed.' % msg, method='POST')