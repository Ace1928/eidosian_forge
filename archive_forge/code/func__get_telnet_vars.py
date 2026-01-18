import binascii
import logging
import os
import pprint
import traceback
from twisted.internet import protocol
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.decorators import defers
from scrapy.utils.engine import print_engine_status
from scrapy.utils.reactor import listen_tcp
from scrapy.utils.trackref import print_live_refs
def _get_telnet_vars(self):
    telnet_vars = {'engine': self.crawler.engine, 'spider': self.crawler.engine.spider, 'slot': self.crawler.engine.slot, 'crawler': self.crawler, 'extensions': self.crawler.extensions, 'stats': self.crawler.stats, 'settings': self.crawler.settings, 'est': lambda: print_engine_status(self.crawler.engine), 'p': pprint.pprint, 'prefs': print_live_refs, 'help': 'This is Scrapy telnet console. For more info see: https://docs.scrapy.org/en/latest/topics/telnetconsole.html'}
    self.crawler.signals.send_catch_log(update_telnet_vars, telnet_vars=telnet_vars)
    return telnet_vars