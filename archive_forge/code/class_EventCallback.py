import unittest
import inspect
import threading
class EventCallback(object):

    def __init__(self, eventName):
        self.eventName = eventName

    def __call__(self, fn):
        fn.event_callback = self.eventName
        return fn