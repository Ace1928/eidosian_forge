from __future__ import absolute_import
import sys
import os
class sysconfig(object):

    @staticmethod
    def get_path(name):
        assert name == 'include'
        return _sysconfig.get_python_inc()
    get_config_var = staticmethod(_sysconfig.get_config_var)