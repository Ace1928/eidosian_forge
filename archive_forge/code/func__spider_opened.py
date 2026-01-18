import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _spider_opened(self, spider):
    self.mindelay = self._min_delay(spider)
    self.maxdelay = self._max_delay(spider)
    spider.download_delay = self._start_delay(spider)