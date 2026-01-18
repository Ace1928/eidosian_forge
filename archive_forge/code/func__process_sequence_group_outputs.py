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
def _process_sequence_group_outputs(self, seq_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:
    prompt_logprobs = outputs.prompt_logprobs
    if prompt_logprobs is not None:
        seq_group.prompt_logprobs = prompt_logprobs
    samples = outputs.samples
    parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    existing_finished_seqs = seq_group.get_finished_seqs()
    parent_child_dict = {parent_seq.seq_id: [] for parent_seq in parent_seqs}
    for sample in samples:
        parent_child_dict[sample.parent_seq_id].append(sample)
    child_seqs: List[Tuple[Sequence, Sequence]] = []
    for parent in parent_seqs:
        child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
        if len(child_samples) == 0:
            parent.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(parent.seq_id)
            self.scheduler.free_seq(parent)
            continue
        for child_sample in child_samples[:-1]:
            new_child_seq_id = next(self.seq_counter)
            child = parent.fork(new_child_seq_id)
            child.append_token_id(child_sample.output_token, child_sample.logprobs)
            child_seqs.append((child, parent))
        last_child_sample = child_samples[-1]
        parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
        child_seqs.append((parent, parent))
    for seq, _ in child_seqs:
        self._decode_sequence(seq, seq_group.sampling_params)
        self._check_stop(seq, seq_group.sampling_params)
    if not seq_group.sampling_params.use_beam_search:
        for seq, parent in child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)
        for seq, parent in child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)
        return
    selected_child_seqs = []
    unselected_child_seqs = []
    beam_width = seq_group.sampling_params.best_of
    length_penalty = seq_group.sampling_params.length_penalty
    existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
    new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs if seq.is_finished()]
    all_finished_seqs = existing_finished_seqs + new_finished_seqs
    all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id), reverse=True)
    for seq, parent, is_new in all_finished_seqs[:beam_width]:
        if is_new:
            selected_child_seqs.append((seq, parent))
    for seq, parent, is_new in all_finished_seqs[beam_width:]:
        if is_new:
            unselected_child_seqs.append((seq, parent))
        else:
            seq_group.remove(seq.seq_id)
    running_child_seqs = [(seq, parent) for seq, parent in child_seqs if not seq.is_finished()]
    running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id), reverse=True)
    if len(running_child_seqs) == 0:
        stop_beam_search = True
    elif len(all_finished_seqs) < beam_width:
        stop_beam_search = False
    else:
        best_running_seq = running_child_seqs[0][0]
        current_worst_seq = all_finished_seqs[beam_width - 1][0]
        stop_beam_search = self._check_beam_search_early_stopping(seq_group.sampling_params.early_stopping, seq_group.sampling_params, best_running_seq, current_worst_seq)
    if stop_beam_search:
        unselected_child_seqs.extend(running_child_seqs)
    else:
        selected_child_seqs.extend(running_child_seqs[:beam_width])
        unselected_child_seqs.extend(running_child_seqs[beam_width:])
    for seq, parent in selected_child_seqs:
        if seq is not parent:
            seq_group.add(seq)
            if not seq.is_finished():
                self.scheduler.fork_seq(parent, seq)
    for seq, parent in selected_child_seqs:
        if seq is parent and seq.is_finished():
            self.scheduler.free_seq(seq)
    for seq, parent in unselected_child_seqs:
        if seq is parent:
            seq_group.remove(seq.seq_id)
            self.scheduler.free_seq(seq)