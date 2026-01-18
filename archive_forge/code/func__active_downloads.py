import hashlib
import logging
from scrapy.utils.misc import create_instance
def _active_downloads(self, slot):
    """Return a number of requests in a Downloader for a given slot"""
    if slot not in self.downloader.slots:
        return 0
    return len(self.downloader.slots[slot].active)