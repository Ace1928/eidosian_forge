import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
def _create_write_queue(self):
    self._write_queue = self._thread_helper.queue()