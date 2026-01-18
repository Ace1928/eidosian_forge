import unittest
import inspect
import threading
class YowLayerEvent:

    def __init__(self, name, **kwargs):
        self.name = name
        self.detached = False
        if 'detached' in kwargs:
            del kwargs['detached']
            self.detached = True
        self.args = kwargs

    def isDetached(self):
        return self.detached

    def getName(self):
        return self.name

    def getArg(self, name):
        return self.args[name] if name in self.args else None