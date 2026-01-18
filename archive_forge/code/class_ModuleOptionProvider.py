from __future__ import (absolute_import, division, print_function)
class ModuleOptionProvider(object):

    def __init__(self, module):
        self.module = module

    def get_option(self, option_name):
        return self.module.params[option_name]