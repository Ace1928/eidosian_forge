from typing import List, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, desc='Processed prompts')
    outputs: List[RequestOutput] = []
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)
    if use_tqdm:
        pbar.close()
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs