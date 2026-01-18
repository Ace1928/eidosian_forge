from __future__ import (absolute_import, division, print_function)
class TemplatedOptionProvider(object):

    def __init__(self, plugin, templar):
        self.plugin = plugin
        self.templar = templar

    def get_option(self, option_name):
        value = self.plugin.get_option(option_name)
        if self.templar.is_template(value):
            value = self.templar.template(variable=value, disable_lookups=False)
        return value