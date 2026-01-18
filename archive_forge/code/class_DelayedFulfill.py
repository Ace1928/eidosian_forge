from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
class DelayedFulfill(Thread):

    def __init__(self, d, p, v):
        self.delay = d
        self.promise = p
        self.value = v
        Thread.__init__(self)

    def run(self):
        sleep(self.delay)
        self.promise.do_resolve(self.value)