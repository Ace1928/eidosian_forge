from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureVariable(Capture):
    names_idx = 0

    def __init__(self, value, ctx):
        if CaptureControl.disabled:
            raise Exception('Attempting to create capture variable with capture off')
        self.ctx = ctx
        self.value = value
        self.name = f'var_{CaptureVariable.names_idx}'
        CaptureVariable.names_idx += 1
        self.ctx['variables'].append(self)

    def __str__(self):
        return self.name

    def execute(self):
        return self.calculated_value

    def apply_ops(self, dataframe):
        self.ctx['variables'][0].calculated_value = dataframe
        for op in self.ctx['operations']:
            op.execute()
        return self.calculated_value