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
@table_function(DATE)
class DateSeries(TableFunction):
    params = ['start', 'stop', 'step_seconds']
    columns = ['date']
    name = 'date_series'

    def initialize(self, start, stop, step_seconds=86400):
        self.start = format_date_time_sqlite(start)
        self.stop = format_date_time_sqlite(stop)
        step_seconds = int(step_seconds)
        self.step_seconds = datetime.timedelta(seconds=step_seconds)
        if self.start.hour == 0 and self.start.minute == 0 and (self.start.second == 0) and (step_seconds >= 86400):
            self.format = '%Y-%m-%d'
        elif self.start.year == 1900 and self.start.month == 1 and (self.start.day == 1) and (self.stop.year == 1900) and (self.stop.month == 1) and (self.stop.day == 1) and (step_seconds < 86400):
            self.format = '%H:%M:%S'
        else:
            self.format = '%Y-%m-%d %H:%M:%S'

    def iterate(self, idx):
        if self.start > self.stop:
            raise StopIteration
        current = self.start
        self.start += self.step_seconds
        return (current.strftime(self.format),)