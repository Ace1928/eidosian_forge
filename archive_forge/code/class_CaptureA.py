from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureA(CaptureF):

    def __str__(self):
        return f'{self.kwargs['name']}'

    def execute(self):
        value = self.kwargs['real_attribute']
        return value