import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@aggregate(DATE)
class avgtdiff(_datetime_heap_agg):

    def finalize(self):
        if self.ct < 1:
            return
        elif self.ct == 1:
            return 0
        total = ct = 0
        dtp = None
        while self.heap:
            if total == 0:
                if dtp is None:
                    dtp = heapq.heappop(self.heap)
                    continue
            dt = heapq.heappop(self.heap)
            diff = dt - dtp
            ct += 1
            total += total_seconds(diff)
            dtp = dt
        return float(total) / ct