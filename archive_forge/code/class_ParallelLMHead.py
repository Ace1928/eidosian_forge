from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.utils import set_weight_attrs
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool=False, params_dtype: Optional[torch.dtype]=None, org_num_embeddings: Optional[int]=None, padding_size: int=DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__(num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size)
        if bias:
            self.bias = Parameter(torch.empty(self.num_embeddings_per_partition, dtype=params_dtype))
            set_weight_attrs(self.bias, {'parallel_dim': 0, 'weight_loader': self.weight_loader})
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")