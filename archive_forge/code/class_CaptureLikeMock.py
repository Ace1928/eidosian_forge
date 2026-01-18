from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureLikeMock:

    def __init__(self, name):
        import unittest.mock as mock
        get_target, attribute = mock._get_target(name)
        self.get_target = get_target
        self.attribute = attribute
        self.name = name

    def __enter__(self):
        self.save = getattr(self.get_target(), self.attribute)
        capt = CaptureA(name=self.name, real_attribute=self.save)
        setattr(self.get_target(), self.attribute, capt)

    def __exit__(self, *exc_info):
        setattr(self.get_target(), self.attribute, self.save)