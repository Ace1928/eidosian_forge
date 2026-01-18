from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
def _ops_str(self):
    res = ''
    for op in self.ctx['operations']:
        if len(res) > 0:
            res += '\n'
        res += str(op)
    return res