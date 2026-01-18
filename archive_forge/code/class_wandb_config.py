from typing import List, Optional
from dataclasses import dataclass, field
@dataclass
class wandb_config:
    project: str = 'llama_recipes'
    entity: Optional[str] = None
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None