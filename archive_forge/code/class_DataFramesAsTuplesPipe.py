import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
@functional_datapipe('_dataframes_as_tuples')
class DataFramesAsTuplesPipe(IterDataPipe):

    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            yield from df_wrapper.iterate(df)