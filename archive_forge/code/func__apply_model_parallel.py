import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def _apply_model_parallel(self, tensor, encoder_output, encoder_mask, incr_state):
    """
        Pipeline application of model parallelism.
        """
    chunks = PipelineHelper.split((tensor, encoder_output, encoder_mask, incr_state))
    work_items = PipelineHelper.schedule_work_items(self.layers, chunks)
    new_incr_state = [{} for _ in chunks]
    for chunk_idx, layer_nos, next_device in work_items:
        s_tensor, s_enc_out, s_enc_mask, s_incr_state = chunks[chunk_idx]
        for layer_no in layer_nos:
            s_tensor, new_incr_state[chunk_idx][layer_no] = self.layers[layer_no](x=s_tensor, encoder_output=s_enc_out, encoder_mask=s_enc_mask, incr_state=s_incr_state.get(layer_no))
        chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_enc_out, s_enc_mask, s_incr_state), next_device)
    tensor_out = PipelineHelper.join([c[0] for c in chunks])
    new_incr_state = PipelineHelper.join(new_incr_state)
    return (tensor_out, new_incr_state)