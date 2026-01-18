from .version_check import check_transformers_version

check_transformers_version()

from .transformers_v4_35_2__configuration_llama import LlamaConfig


class DeciLMConfig(LlamaConfig):
    r"""
    Args:
        num_key_value_heads_per_layer (`List[int]`):
            The number of key-value heads per layer.
    """
    model_type = "deci"

    def __init__(
            self,
            num_key_value_heads_per_layer: list = None,
            **kwargs,
    ):
        self.num_key_value_heads_per_layer = num_key_value_heads_per_layer
        super().__init__(**kwargs)
