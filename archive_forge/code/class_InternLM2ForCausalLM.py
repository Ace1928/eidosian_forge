from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class InternLM2ForCausalLM(nn.Module):

    def __init__(self, config: PretrainedConfig, linear_method: Optional[LinearMethodBase]=None) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = InternLM2Model(config, linear_method)
        self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.output.weight, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None):
        stacked_params_mapping = [('gate_up_proj', 'w1', 0), ('gate_up_proj', 'w3', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision):
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                if 'wqkv' in name:
                    config = self.config
                    kv_groups = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(-1, 2 + kv_groups, head_dim, loaded_weight.shape[-1])
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1], dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, 'q')
                    weight_loader(param, wk, 'k')
                    weight_loader(param, wv, 'v')
                else:
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)