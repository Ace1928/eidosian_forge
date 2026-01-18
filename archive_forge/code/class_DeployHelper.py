from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class DeployHelper(object):

    def __init__(self, module):
        self.module = module
        self.file_args = module.load_file_common_arguments(module.params)
        self.clean = module.params['clean']
        self.current_path = module.params['current_path']
        self.keep_releases = module.params['keep_releases']
        self.path = module.params['path']
        self.release = module.params['release']
        self.releases_path = module.params['releases_path']
        self.shared_path = module.params['shared_path']
        self.state = module.params['state']
        self.unfinished_filename = module.params['unfinished_filename']

    def gather_facts(self):
        current_path = os.path.join(self.path, self.current_path)
        releases_path = os.path.join(self.path, self.releases_path)
        if self.shared_path:
            shared_path = os.path.join(self.path, self.shared_path)
        else:
            shared_path = None
        previous_release, previous_release_path = self._get_last_release(current_path)
        if not self.release and (self.state == 'query' or self.state == 'present'):
            self.release = time.strftime('%Y%m%d%H%M%S')
        if self.release:
            new_release_path = os.path.join(releases_path, self.release)
        else:
            new_release_path = None
        return {'project_path': self.path, 'current_path': current_path, 'releases_path': releases_path, 'shared_path': shared_path, 'previous_release': previous_release, 'previous_release_path': previous_release_path, 'new_release': self.release, 'new_release_path': new_release_path, 'unfinished_filename': self.unfinished_filename}

    def delete_path(self, path):
        if not os.path.lexists(path):
            return False
        if not os.path.isdir(path):
            self.module.fail_json(msg='%s exists but is not a directory' % path)
        if not self.module.check_mode:
            try:
                shutil.rmtree(path, ignore_errors=False)
            except Exception as e:
                self.module.fail_json(msg='rmtree failed: %s' % to_native(e), exception=traceback.format_exc())
        return True

    def create_path(self, path):
        changed = False
        if not os.path.lexists(path):
            changed = True
            if not self.module.check_mode:
                os.makedirs(path)
        elif not os.path.isdir(path):
            self.module.fail_json(msg='%s exists but is not a directory' % path)
        changed += self.module.set_directory_attributes_if_different(self._get_file_args(path), changed)
        return changed

    def check_link(self, path):
        if os.path.lexists(path):
            if not os.path.islink(path):
                self.module.fail_json(msg='%s exists but is not a symbolic link' % path)

    def create_link(self, source, link_name):
        if os.path.islink(link_name):
            norm_link = os.path.normpath(os.path.realpath(link_name))
            norm_source = os.path.normpath(os.path.realpath(source))
            if norm_link == norm_source:
                changed = False
            else:
                changed = True
                if not self.module.check_mode:
                    if not os.path.lexists(source):
                        self.module.fail_json(msg="the symlink target %s doesn't exists" % source)
                    tmp_link_name = link_name + '.' + self.unfinished_filename
                    if os.path.islink(tmp_link_name):
                        os.unlink(tmp_link_name)
                    os.symlink(source, tmp_link_name)
                    os.rename(tmp_link_name, link_name)
        else:
            changed = True
            if not self.module.check_mode:
                os.symlink(source, link_name)
        return changed

    def remove_unfinished_file(self, new_release_path):
        changed = False
        unfinished_file_path = os.path.join(new_release_path, self.unfinished_filename)
        if os.path.lexists(unfinished_file_path):
            changed = True
            if not self.module.check_mode:
                os.remove(unfinished_file_path)
        return changed

    def remove_unfinished_builds(self, releases_path):
        changes = 0
        for release in os.listdir(releases_path):
            if os.path.isfile(os.path.join(releases_path, release, self.unfinished_filename)):
                if self.module.check_mode:
                    changes += 1
                else:
                    changes += self.delete_path(os.path.join(releases_path, release))
        return changes

    def remove_unfinished_link(self, path):
        changed = False
        if not self.release:
            return changed
        tmp_link_name = os.path.join(path, self.release + '.' + self.unfinished_filename)
        if not self.module.check_mode and os.path.exists(tmp_link_name):
            changed = True
            os.remove(tmp_link_name)
        return changed

    def cleanup(self, releases_path, reserve_version):
        changes = 0
        if os.path.lexists(releases_path):
            releases = [f for f in os.listdir(releases_path) if os.path.isdir(os.path.join(releases_path, f))]
            try:
                releases.remove(reserve_version)
            except ValueError:
                pass
            if not self.module.check_mode:
                releases.sort(key=lambda x: os.path.getctime(os.path.join(releases_path, x)), reverse=True)
                for release in releases[self.keep_releases:]:
                    changes += self.delete_path(os.path.join(releases_path, release))
            elif len(releases) > self.keep_releases:
                changes += len(releases) - self.keep_releases
        return changes

    def _get_file_args(self, path):
        file_args = self.file_args.copy()
        file_args['path'] = path
        return file_args

    def _get_last_release(self, current_path):
        previous_release = None
        previous_release_path = None
        if os.path.lexists(current_path):
            previous_release_path = os.path.realpath(current_path)
            previous_release = os.path.basename(previous_release_path)
        return (previous_release, previous_release_path)