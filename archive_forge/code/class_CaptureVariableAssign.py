from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureVariableAssign(CaptureF):

    def __str__(self):
        variable = self.kwargs['variable']
        value = self.kwargs['value']
        return f'{variable} = {value}'

    def execute(self):
        self.kwargs['variable'].calculated_value = self.kwargs['value'].execute()