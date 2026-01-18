import unittest
import inspect
import threading
class YowLayerInterface(object):

    def __init__(self, layer):
        self._layer = layer