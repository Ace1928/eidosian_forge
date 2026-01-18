import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
@functional_datapipe('_to_dataframes_pipe', enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):

    def __init__(self, source_datapipe, dataframe_size=10, columns=None):
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        try:
            return list(item)
        except Exception:
            return [item]

    def __iter__(self):
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)