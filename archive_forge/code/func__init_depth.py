import logging
from scrapy.http import Request
def _init_depth(self, response, spider):
    if 'depth' not in response.meta:
        response.meta['depth'] = 0
        if self.verbose_stats:
            self.stats.inc_value('request_depth_count/0', spider=spider)