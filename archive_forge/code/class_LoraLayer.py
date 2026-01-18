import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
class LoraLayer(BaseTunerLayer):
    adapter_layer_names = ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B')
    other_param_names = ('r', 'lora_alpha', 'scaling', 'lora_dropout')

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = (base_layer.in_features, base_layer.out_features)
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = (base_layer.in_channels, base_layer.out_channels)
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = (base_layer.num_embeddings, base_layer.embedding_dim)
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.ds_shape if hasattr(base_layer.weight, 'ds_shape') else base_layer.weight.shape
        elif hasattr(base_layer, 'infeatures') and hasattr(base_layer, 'outfeatures'):
            in_features, out_features = (base_layer.infeatures, base_layer.outfeatures)
        elif hasattr(base_layer, 'input_size') and hasattr(base_layer, 'output_size'):
            in_features, out_features = (base_layer.input_size, base_layer.output_size)
        elif hasattr(base_layer, 'codebooks') and base_layer.__class__.__name__ == 'QuantizedLinear':
            in_features, out_features = (base_layer.in_features, base_layer.out_features)
        elif hasattr(base_layer, 'w_bit') and base_layer.__class__.__name__ == 'WQLinear_GEMM':
            in_features, out_features = (base_layer.in_features, base_layer.out_features)
        else:
            raise ValueError(f'Unsupported layer type {type(base_layer)}')
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool=False):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == 'loftq':
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        for weight_name in ('weight', 'qweight'):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == 'gaussian':
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f'Unknown initialization init_lora_weights={init_lora_weights!r}')
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init
        weight = self.get_base_layer().weight
        kwargs = {'num_bits': self.kwargs.get('loftq_bits', 4), 'reduced_rank': self.r[adapter_name], 'num_iter': self.kwargs.get('loftq_iter', 1)}
        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def dora_init(self, adapter_name: str) -> None:
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        scaling = self.scaling[adapter_name]
        with gather_params_ctx(self.get_base_layer()):
            weight = self.get_base_layer().weight
            lora_weight = lora_B.weight @ lora_A.weight
            weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        self.lora_magnitude_vector = nn.ParameterDict()
        self.lora_magnitude_vector[adapter_name] = nn.Parameter(weight_norm, requires_grad=True)
        self.adapter_layer_names = self.adapter_layer_names[:] + ('lora_magnitude_vector',)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = lora_B.weight @ lora_A.weight
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight = self.get_base_layer().weight
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * F.linear(x, transpose(weight, self.fan_in_fan_out)) + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale