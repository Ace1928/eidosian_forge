import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
def _as_list(self, item):
    try:
        return list(item)
    except Exception:
        return [item]