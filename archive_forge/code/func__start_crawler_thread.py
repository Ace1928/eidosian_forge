from argparse import Namespace
from threading import Thread
from typing import List, Type
from scrapy import Spider
from scrapy.commands import ScrapyCommand
from scrapy.http import Request
from scrapy.shell import Shell
from scrapy.utils.spider import DefaultSpider, spidercls_for_request
from scrapy.utils.url import guess_scheme
def _start_crawler_thread(self):
    t = Thread(target=self.crawler_process.start, kwargs={'stop_after_crawl': False, 'install_signal_handlers': False})
    t.daemon = True
    t.start()