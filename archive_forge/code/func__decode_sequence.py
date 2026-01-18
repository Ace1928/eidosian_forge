import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
    """Decodes the new token for a sequence."""
    new_tokens, new_output_text, prefix_offset, read_offset = detokenize_incrementally(self.get_tokenizer_for_seq(seq), all_input_ids=seq.get_token_ids(), prev_tokens=seq.tokens, prefix_offset=seq.prefix_offset, read_offset=seq.read_offset, skip_special_tokens=prms.skip_special_tokens, spaces_between_special_tokens=prms.spaces_between_special_tokens)
    if seq.tokens is None:
        seq.tokens = new_tokens
    else:
        seq.tokens.extend(new_tokens)
    seq.prefix_offset = prefix_offset
    seq.read_offset = read_offset
    seq.output_text += new_output_text