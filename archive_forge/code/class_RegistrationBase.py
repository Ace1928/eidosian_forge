from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
class RegistrationBase(object):
    """
    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 10.0.0.
    There is no replacement for it; please contact the community.general
    maintainers in case you are using it.
    """

    def __init__(self, module, username=None, password=None):
        self.module = module
        self.username = username
        self.password = password

    def configure(self):
        raise NotImplementedError('Must be implemented by a sub-class')

    def enable(self):
        redhat_repo = '/etc/yum.repos.d/redhat.repo'
        if os.path.isfile(redhat_repo):
            os.unlink(redhat_repo)

    def register(self):
        raise NotImplementedError('Must be implemented by a sub-class')

    def unregister(self):
        raise NotImplementedError('Must be implemented by a sub-class')

    def unsubscribe(self):
        raise NotImplementedError('Must be implemented by a sub-class')

    def update_plugin_conf(self, plugin, enabled=True):
        plugin_conf = '/etc/yum/pluginconf.d/%s.conf' % plugin
        if os.path.isfile(plugin_conf):
            tmpfd, tmpfile = tempfile.mkstemp()
            shutil.copy2(plugin_conf, tmpfile)
            cfg = configparser.ConfigParser()
            cfg.read([tmpfile])
            if enabled:
                cfg.set('main', 'enabled', 1)
            else:
                cfg.set('main', 'enabled', 0)
            fd = open(tmpfile, 'w+')
            cfg.write(fd)
            fd.close()
            self.module.atomic_move(tmpfile, plugin_conf)

    def subscribe(self, **kwargs):
        raise NotImplementedError('Must be implemented by a sub-class')