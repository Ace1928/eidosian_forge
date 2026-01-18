import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
def _chat_completion_turns(self, dialogs: List[Dialog], temperature: float=0.6, top_p: float=0.9, max_gen_len: Optional[int]=None, logprobs: bool=False) -> List[ChatPrediction]:
    if self.tokenizer.step_id is None:
        raise RuntimeError('Model not suitable for chat_completion_step()')
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(any([tag in msg['content'] for tag in SPECIAL_TAGS for msg in dialog]))
        if dialog[0]['role'] != 'system':
            dialog = [{'role': 'system', 'content': ''}] + dialog
        dialog_tokens = dialog_prompt_tokens(self.tokenizer, dialog)
        prompt_tokens.append(dialog_tokens)
    generation_tokens, generation_logprobs = self.generate(prompt_tokens=prompt_tokens, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs, stop_token=self.tokenizer.step_id)
    if logprobs:
        assert generation_logprobs is not None
        return [{'generation': {'role': 'assistant', 'destination': 'user', 'content': self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR}, 'tokens': [self.tokenizer.token_piece(x) for x in t], 'logprobs': logprobs_i} for t, logprobs_i, unsafe in zip(generation_tokens, generation_logprobs, unsafe_requests)]
    return [{'generation': {'role': 'assistant', 'destination': 'user', 'content': self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR}} for t, unsafe in zip(generation_tokens, unsafe_requests)]