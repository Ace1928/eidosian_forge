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
def _check_beam_search_early_stopping(self, early_stopping: Union[bool, str], sampling_params: SamplingParams, best_running_seq: Sequence, current_worst_seq: Sequence) -> bool:
    assert sampling_params.use_beam_search
    length_penalty = sampling_params.length_penalty
    if early_stopping is True:
        return True
    current_worst_score = current_worst_seq.get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(current_worst_seq).eos_token_id)
    if early_stopping is False:
        highest_attainable_score = best_running_seq.get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id)
    else:
        assert early_stopping == 'never'
        if length_penalty > 0.0:
            max_possible_length = max(best_running_seq.get_prompt_len() + sampling_params.max_tokens, self.scheduler_config.max_model_len)
            highest_attainable_score = best_running_seq.get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id, seq_len=max_possible_length)
        else:
            highest_attainable_score = best_running_seq.get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id)
    return current_worst_score >= highest_attainable_score