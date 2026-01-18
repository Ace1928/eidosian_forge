from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class DataFrameTracedOps(DFIterDataPipe):

    def __init__(self, source_datapipe, output_var):
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)