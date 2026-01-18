import curses
import sys
import threading
from datetime import datetime
from itertools import count
from math import ceil
from textwrap import wrap
from time import time
from celery import VERSION_BANNER, states
from celery.app import app_or_default
from celery.utils.text import abbr, abbrtask
class DisplayThread(threading.Thread):

    def __init__(self, display):
        self.display = display
        self.shutdown = False
        super().__init__()

    def run(self):
        while not self.shutdown:
            self.display.draw()
            self.display.nap()