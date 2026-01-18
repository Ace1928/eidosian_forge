import logging
import socket
import sys
from importlib import import_module
from pprint import pformat
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.mail import MailSender
from scrapy.utils.engine import get_engine_status
def _check_warning(self):
    if self.warned:
        return
    if self.get_virtual_size() > self.warning:
        self.crawler.stats.set_value('memusage/warning_reached', 1)
        mem = self.warning / 1024 / 1024
        logger.warning('Memory usage reached %(memusage)dMiB', {'memusage': mem}, extra={'crawler': self.crawler})
        if self.notify_mails:
            subj = f'{self.crawler.settings['BOT_NAME']} warning: memory usage reached {mem}MiB at {socket.gethostname()}'
            self._send_report(self.notify_mails, subj)
            self.crawler.stats.set_value('memusage/warning_notified', 1)
        self.warned = True