from typing import List, Optional
import time
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SequenceGroup,
from vllm.lora.request import LoRARequest
class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(self, request_id: str, prompt: str, prompt_token_ids: List[int], prompt_logprobs: Optional[PromptLogprobs], outputs: List[CompletionOutput], finished: bool, metrics: Optional[RequestMetrics]=None, lora_request: Optional[LoRARequest]=None) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> 'RequestOutput':
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        if seq_group.sampling_params.use_beam_search:
            sorting_key = lambda seq: seq.get_beam_search_score(seq_group.sampling_params.length_penalty)
        else:
            sorting_key = lambda seq: seq.get_cumulative_logprob()
        sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
        top_n_seqs = sorted_seqs[:n]
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs is None:
                logprobs = None
            finshed_reason = SequenceStatus.get_finished_reason(seq.status)
            output = CompletionOutput(seqs.index(seq), seq.output_text, seq.get_output_token_ids(), seq.get_cumulative_logprob(), logprobs, finshed_reason)
            outputs.append(output)
        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
        finished = seq_group.is_finished()
        finished_time = time.time() if finished else None
        seq_group.set_finished_time(finished_time)
        return cls(seq_group.request_id, prompt, prompt_token_ids, prompt_logprobs, outputs, finished, seq_group.metrics, lora_request=seq_group.lora_request)

    def __repr__(self) -> str:
        return f'RequestOutput(request_id={self.request_id}, prompt={self.prompt!r}, prompt_token_ids={self.prompt_token_ids}, prompt_logprobs={self.prompt_logprobs}, outputs={self.outputs}, finished={self.finished}, metrics={self.metrics}, lora_request={self.lora_request})'