from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureCall(Capture):

    def __init__(self, callable, ctx=None, **kwargs):
        if ctx is None:
            self.ctx = {'operations': [], 'variables': []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        self.callable = callable

    def __str__(self):
        return '{callable}({args},{kwargs})'.format(callable=self.callable, **self.kwargs)

    def execute(self):
        executed_args = []
        for arg in self.kwargs['args']:
            if isinstance(arg, Capture):
                executed_args.append(arg.execute())
            else:
                executed_args.append(arg)
        left = get_val(self.callable)
        return left(*executed_args, **self.kwargs['kwargs'])