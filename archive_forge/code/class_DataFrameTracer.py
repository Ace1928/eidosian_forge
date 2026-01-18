from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
@functional_datapipe('trace_as_dataframe')
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):
    source_datapipe = None

    def set_shuffle_settings(self, *args, **kwargs):
        pass

    def is_shardable(self):
        return False

    def __init__(self, source_datapipe, schema_df=None):
        self.source_datapipe = source_datapipe
        if schema_df is None:
            schema_df = next(iter(self.source_datapipe))
        super().__init__(schema_df=schema_df)