from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureF(Capture):

    def __init__(self, ctx=None, **kwargs):
        if ctx is None:
            self.ctx = {'operations': [], 'variables': []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs