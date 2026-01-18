from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
class PrefixPool:
    """Manages all the prompt prefixes.

    NOTE: This feature is experimental and may be replaced with automatic
        prefix caching in the future.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
    """

    def __init__(self, block_size: int) -> None:
        self.prefixes: Dict[int, Prefix] = {}
        self.block_size = block_size

    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        return tuple(token_ids[:new_length])

    def add_or_get_prefix(self, token_ids: Sequence[int], lora_int_id: int) -> Optional[Prefix]:
        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            return None
        prefix = Prefix(token_ids, self.block_size)
        prefix_hash = hash((prefix, lora_int_id))
        if prefix_hash not in self.prefixes:
            self.prefixes[prefix_hash] = prefix
        return self.prefixes[prefix_hash]