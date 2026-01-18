from __future__ import (absolute_import, division, print_function)
import os
import os.path
import random
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import text_type, string_types
from ansible.plugins import AnsiblePlugin
class ShellBase(AnsiblePlugin):

    def __init__(self):
        super(ShellBase, self).__init__()
        self.env = {}
        self.tmpdir = None
        self.executable = None

    def _normalize_system_tmpdirs(self):
        normalized_paths = [d.rstrip('/') for d in self.get_option('system_tmpdirs')]
        if not all((os.path.isabs(d) for d in normalized_paths)):
            raise AnsibleError('The configured system_tmpdirs contains a relative path: {0}. All system_tmpdirs must be absolute'.format(to_native(normalized_paths)))
        self.set_option('system_tmpdirs', normalized_paths)

    def set_options(self, task_keys=None, var_options=None, direct=None):
        super(ShellBase, self).set_options(task_keys=task_keys, var_options=var_options, direct=direct)
        env = self.get_option('environment')
        if isinstance(env, string_types):
            raise AnsibleError('The "envirionment" keyword takes a list of dictionaries or a dictionary, not a string')
        if not isinstance(env, Sequence):
            env = [env]
        for env_dict in env:
            if not isinstance(env_dict, Mapping):
                raise AnsibleError('The "envirionment" keyword takes a list of dictionaries (or single dictionary), but got a "%s" instead' % type(env_dict))
            self.env.update(env_dict)
        try:
            self._normalize_system_tmpdirs()
        except KeyError:
            pass

    @staticmethod
    def _generate_temp_dir_name():
        return 'ansible-tmp-%s-%s-%s' % (time.time(), os.getpid(), random.randint(0, 2 ** 48))

    def env_prefix(self, **kwargs):
        return ' '.join(['%s=%s' % (k, shlex.quote(text_type(v))) for k, v in kwargs.items()])

    def join_path(self, *args):
        return os.path.join(*args)

    def get_remote_filename(self, pathname):
        base_name = os.path.basename(pathname.strip())
        return base_name.strip()

    def path_has_trailing_slash(self, path):
        return path.endswith('/')

    def chmod(self, paths, mode):
        cmd = ['chmod', mode]
        cmd.extend(paths)
        cmd = [shlex.quote(c) for c in cmd]
        return ' '.join(cmd)

    def chown(self, paths, user):
        cmd = ['chown', user]
        cmd.extend(paths)
        cmd = [shlex.quote(c) for c in cmd]
        return ' '.join(cmd)

    def chgrp(self, paths, group):
        cmd = ['chgrp', group]
        cmd.extend(paths)
        cmd = [shlex.quote(c) for c in cmd]
        return ' '.join(cmd)

    def set_user_facl(self, paths, user, mode):
        """Only sets acls for users as that's really all we need"""
        cmd = ['setfacl', '-m', 'u:%s:%s' % (user, mode)]
        cmd.extend(paths)
        cmd = [shlex.quote(c) for c in cmd]
        return ' '.join(cmd)

    def remove(self, path, recurse=False):
        path = shlex.quote(path)
        cmd = 'rm -f '
        if recurse:
            cmd += '-r '
        return cmd + '%s %s' % (path, self._SHELL_REDIRECT_ALLNULL)

    def exists(self, path):
        cmd = ['test', '-e', shlex.quote(path)]
        return ' '.join(cmd)

    def mkdtemp(self, basefile=None, system=False, mode=448, tmpdir=None):
        if not basefile:
            basefile = self.__class__._generate_temp_dir_name()
        if system:
            if tmpdir:
                tmpdir = tmpdir.rstrip('/')
            if tmpdir in self.get_option('system_tmpdirs'):
                basetmpdir = tmpdir
            else:
                basetmpdir = self.get_option('system_tmpdirs')[0]
        elif tmpdir is None:
            basetmpdir = self.get_option('remote_tmp')
        else:
            basetmpdir = tmpdir
        basetmp = self.join_path(basetmpdir, basefile)
        cmd = 'mkdir -p %s echo %s %s' % (self._SHELL_SUB_LEFT, basetmpdir, self._SHELL_SUB_RIGHT)
        cmd += '%s mkdir %s echo %s %s' % (self._SHELL_AND, self._SHELL_SUB_LEFT, basetmp, self._SHELL_SUB_RIGHT)
        cmd += ' %s echo %s=%s echo %s %s' % (self._SHELL_AND, basefile, self._SHELL_SUB_LEFT, basetmp, self._SHELL_SUB_RIGHT)
        if mode:
            tmp_umask = 511 & ~mode
            cmd = '%s umask %o %s %s %s' % (self._SHELL_GROUP_LEFT, tmp_umask, self._SHELL_AND, cmd, self._SHELL_GROUP_RIGHT)
        return cmd

    def expand_user(self, user_home_path, username=''):
        """ Return a command to expand tildes in a path

        It can be either "~" or "~username". We just ignore $HOME
        We use the POSIX definition of a username:
            http://pubs.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap03.html#tag_03_426
            http://pubs.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap03.html#tag_03_276

            Falls back to 'current working directory' as we assume 'home is where the remote user ends up'
        """
        if user_home_path != '~':
            if not _USER_HOME_PATH_RE.match(user_home_path):
                user_home_path = shlex.quote(user_home_path)
        elif username:
            user_home_path += username
        return 'echo %s' % user_home_path

    def pwd(self):
        """Return the working directory after connecting"""
        return 'echo %spwd%s' % (self._SHELL_SUB_LEFT, self._SHELL_SUB_RIGHT)

    def build_module_command(self, env_string, shebang, cmd, arg_path=None):
        if cmd.strip() != '':
            cmd = shlex.quote(cmd)
        cmd_parts = []
        if shebang:
            shebang = shebang.replace('#!', '').strip()
        else:
            shebang = ''
        cmd_parts.extend([env_string.strip(), shebang, cmd])
        if arg_path is not None:
            cmd_parts.append(arg_path)
        new_cmd = ' '.join(cmd_parts)
        return new_cmd

    def append_command(self, cmd, cmd_to_append):
        """Append an additional command if supported by the shell"""
        if self._SHELL_AND:
            cmd += ' %s %s' % (self._SHELL_AND, cmd_to_append)
        return cmd

    def wrap_for_exec(self, cmd):
        """wrap script execution with any necessary decoration (eg '&' for quoted powershell script paths)"""
        return cmd

    def quote(self, cmd):
        """Returns a shell-escaped string that can be safely used as one token in a shell command line"""
        return shlex.quote(cmd)