import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray', action='store_true', help='use Ray to start the LLM engine in a separate process as the server process.')
        parser.add_argument('--disable-log-requests', action='store_true', help='disable logging requests')
        parser.add_argument('--max-log-len', type=int, default=None, help='max number of prompt characters or prompt ID numbers being printed in log. Default: unlimited.')
        return parser