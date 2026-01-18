from typing import Any, Dict
import torch
from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once
def _get_context_batch_size(self, **ctxt_inputs: torch.Tensor) -> int:
    """
        Return the batch size of the context.
        """
    if ctxt_inputs['ctxt_tokens'] is not None:
        return ctxt_inputs['ctxt_tokens'].size(0)
    else:
        return ctxt_inputs['ctxt_image'].size(0)