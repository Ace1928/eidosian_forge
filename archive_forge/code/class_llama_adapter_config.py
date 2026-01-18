from dataclasses import dataclass, field
from typing import List
@dataclass
class llama_adapter_config:
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = 'CAUSAL_LM'