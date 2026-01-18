from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):

    def as_datapipe(self):
        return DataFrameTracedOps(self.ctx['variables'][0].source_datapipe, self)

    def raw_iterator(self):
        return self.as_datapipe().__iter__()

    def __iter__(self):
        return iter(self._dataframes_as_tuples())

    def batch(self, batch_size=10, drop_last: bool=False, wrapper_class=DataChunkDF):
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        dp._dp_contains_dataframe = True
        return dp

    def groupby(self, group_key_fn, *, buffer_size=10000, group_size=None, guaranteed_group_size=None, drop_remaining=False):
        dp = self._dataframes_per_row()
        dp = dp.as_datapipe().groupby(group_key_fn, buffer_size=buffer_size, group_size=group_size, guaranteed_group_size=guaranteed_group_size, drop_remaining=drop_remaining)
        return dp

    def shuffle(self, *args, **kwargs):
        return self._dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self._dataframes_filter(*args, **kwargs)

    def collate(self, *args, **kwargs):
        raise Exception("Can't collate unbatched DataFrames stream")

    def __getattr__(self, attrname):
        if attrname in UNIMPLEMENTED_ATTR:
            raise AttributeError('Attempting to get ', attrname)
        if attrname in DATAPIPES_OPS:
            return self.as_datapipe().__getattr__(attrname)
        return super().__getattr__(attrname)