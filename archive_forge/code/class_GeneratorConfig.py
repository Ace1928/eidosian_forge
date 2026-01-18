from dataclasses import dataclass
from typing import Callable, Optional
import datasets
@dataclass
class GeneratorConfig(datasets.BuilderConfig):
    generator: Optional[Callable] = None
    gen_kwargs: Optional[dict] = None
    features: Optional[datasets.Features] = None

    def __post_init__(self):
        assert self.generator is not None, 'generator must be specified'
        if self.gen_kwargs is None:
            self.gen_kwargs = {}