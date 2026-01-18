import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
@functional_datapipe('_dataframes_concat', enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe, batch=3):
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)