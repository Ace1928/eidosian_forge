import logging
from pyomo.common import Factory
from pyomo.common.plugin_base import PluginError
class UnknownDataManager(object):

    def __init__(self, *args, **kwds):
        self.type = kwds['type']

    def available(self):
        return False