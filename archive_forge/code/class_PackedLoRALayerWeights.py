from typing import List, Optional
import torch
from vllm.utils import in_wsl
class PackedLoRALayerWeights(LoRALayerWeights):
    """LoRA used for packed layers (eg. qkv_proj)."""

    def __init__(self, module_name: str, rank: int, lora_alphas: List[int], lora_a: List[torch.Tensor], lora_b: List[torch.Tensor], scaling: Optional[List[float]]=None) -> None:
        super().__init__(module_name=module_name, rank=rank, lora_alpha=0, lora_a=lora_a, lora_b=lora_b, scaling=scaling, embeddings_tensor=None)
        self.lora_alphas = lora_alphas
        if scaling is None:
            self.scaling = [lora_alpha / self.rank for lora_alpha in self.lora_alphas]

    @classmethod
    def pack(cls, loras: List['LoRALayerWeights']) -> 'PackedLoRALayerWeights':
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_lora = next((lora for lora in loras if lora is not None))
        for lora in loras:
            if lora is None:
                continue
            lora.optimize()
        rank = first_lora.rank
        module_name = first_lora.module_name
        obj = cls(module_name, rank, [lora.lora_alpha if lora is not None else None for lora in loras], [lora.lora_a if lora is not None else None for lora in loras], [lora.lora_b if lora is not None else None for lora in loras], scaling=[1 if lora is not None else None for lora in loras])
        return obj

    def optimize(self) -> 'PackedLoRALayerWeights':
        """Optimize the LoRA by merging the scaling into lora_b."""
        for i in range(len(self.lora_b)):
            if self.scaling[i] == 1 or self.lora_b[i] is None:
                continue
            self.lora_b[i] *= self.scaling[i]
            self.scaling[i] = 1
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True