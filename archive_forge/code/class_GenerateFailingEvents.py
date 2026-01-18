import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
class GenerateFailingEvents(HasTraits):
    name = Str

    def _name_changed(self):
        raise RuntimeError