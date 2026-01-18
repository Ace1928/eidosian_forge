from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
class FakeThenPromise:

    def __init__(self, raises=True):
        self.raises = raises

    def then(self, s=None, f=None):
        if self.raises:
            raise Exception("FakeThenPromise raises in 'then'")