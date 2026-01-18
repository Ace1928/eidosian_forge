import configparser
import logging
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _, _LE
class SwiftConfigParser(configparser.ConfigParser):

    def get(self, *args, **kwargs):
        value = super(configparser.ConfigParser, self).get(*args, **kwargs)
        return self._process_quotes(value)

    @staticmethod
    def _process_quotes(value):
        if value:
            if value[0] in '"\'':
                if len(value) == 1 or value[-1] != value[0]:
                    raise ValueError('Non-closed quote: %s' % value)
                value = value[1:-1]
        return value