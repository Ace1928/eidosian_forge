from ...utils import split_and_load
from .... import autograd
def _get_data_and_label(self, batch, ctx, batch_axis=0):
    data = batch[0]
    label = batch[1]
    data = split_and_load(data, ctx_list=ctx, batch_axis=batch_axis)
    label = split_and_load(label, ctx_list=ctx, batch_axis=batch_axis)
    return (data, label)