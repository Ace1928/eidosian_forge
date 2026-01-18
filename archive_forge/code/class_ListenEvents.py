import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
class ListenEvents(HasTraits):

    def _name_changed(self, object, name, old, new):
        events['_name_changed'] = (name, old, new)

    def _age_changed(self, object, name, old, new):
        events['_age_changed'] = (name, old, new)

    def _weight_changed(self, object, name, old, new):
        events['_weight_changed'] = (name, old, new)

    def alt_name_changed(self, object, name, old, new):
        events['alt_name_changed'] = (name, old, new)

    def alt_weight_changed(self, object, name, old, new):
        events['alt_weight_changed'] = (name, old, new)