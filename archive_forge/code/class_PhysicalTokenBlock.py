from typing import List
from vllm.utils import Device
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(self, device: Device, block_number: int, block_size: int) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.ref_count = 0

    def __repr__(self) -> str:
        return f'PhysicalTokenBlock(device={self.device}, block_number={self.block_number}, ref_count={self.ref_count})'