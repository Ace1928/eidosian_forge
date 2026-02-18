from __future__ import annotations
import asyncio
from typing import Iterator, List, Optional
from ..core.model import ModelInterface
from ..models import ModelConfig
from llm_forge.engine.local_cli import LocalCLIEngine, EngineConfig
from eidosian_core import eidosian

class LlmForgeModel(ModelInterface):
    """Adapter for llm_forge engines within agent_forge."""

    def __init__(self, config: ModelConfig):
        self.config = config
        # Map agent_forge config to llm_forge config
        self.engine_config = EngineConfig(
            model_path=config.model_name,
            ctx_size=config.max_context,
            temp=config.temperature,
            n_predict=config.max_tokens
        )
        self.engine = LocalCLIEngine(self.engine_config)

    @eidosian()
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # Agent Forge is mostly synchronous, we wrap the async call
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.engine.generate(prompt, n_predict=max_tokens or self.config.max_tokens)
        )

    @eidosian()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        # LocalCLIEngine doesn't support streaming yet in current implementation
        response = self.generate(prompt, max_tokens, temperature)
        yield response

    @eidosian()
    def tokenize(self, text: str) -> List[int]:
        return [ord(c) for c in text] # Fallback until llm_forge exposes tokenizer

    @eidosian()
    def num_tokens(self, text: str) -> int:
        return len(text)
