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
@aggregate(MATH)
class avgrange(_heap_agg):

    def finalize(self):
        if self.ct == 0:
            return
        elif self.ct == 1:
            return 0
        total = ct = 0
        prev = None
        while self.heap:
            if total == 0:
                if prev is None:
                    prev = heapq.heappop(self.heap)
                    continue
            curr = heapq.heappop(self.heap)
            diff = curr - prev
            ct += 1
            total += diff
            prev = curr
        return float(total) / ct