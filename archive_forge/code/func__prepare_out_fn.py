from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
def _prepare_out_fn(self, outputs, device_mesh):
    prepared_outputs = []
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for out, out_layout, desired_out_layout in zip(outputs, self.output_layouts, self.desired_output_layouts):
        if out_layout is not None:
            if isinstance(out, DTensor):
                assert out.placements[0] == out_layout
                dt_out = out
            else:
                dt_out = DTensor.from_local(out, device_mesh, (out_layout,), run_check=False)
            if out_layout != desired_out_layout:
                dt_out = dt_out.redistribute(placements=(desired_out_layout,))
            prepared_outputs.append(dt_out.to_local() if self.use_local_output else dt_out)
        else:
            prepared_outputs.append(out)
    if len(prepared_outputs) == 1:
        return prepared_outputs[0]
    else:
        return tuple(prepared_outputs)