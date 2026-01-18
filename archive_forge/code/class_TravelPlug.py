from abc import ABC
from traits.adaptation.api import PurePythonAdapter as Adapter
class TravelPlug(object):

    def __init__(self, mode):
        self.mode = mode