import hashlib
import logging
from scrapy.utils.misc import create_instance
class DownloaderInterface:

    def __init__(self, crawler):
        self.downloader = crawler.engine.downloader

    def stats(self, possible_slots):
        return [(self._active_downloads(slot), slot) for slot in possible_slots]

    def get_slot_key(self, request):
        return self.downloader._get_slot_key(request, None)

    def _active_downloads(self, slot):
        """Return a number of requests in a Downloader for a given slot"""
        if slot not in self.downloader.slots:
            return 0
        return len(self.downloader.slots[slot].active)