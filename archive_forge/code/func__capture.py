import struct
from oslo_log import log as logging
def _capture(self, chunk, only=None):
    for name, region in self._capture_regions.items():
        if only and name not in only:
            continue
        if not region.complete:
            region.capture(chunk, self._total_count)