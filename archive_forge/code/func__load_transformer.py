import importlib.util
import logging
from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.self_hosted import SelfHostedPipeline
from langchain_community.llms.utils import enforce_stop_tokens
def _load_transformer(model_id: str=DEFAULT_MODEL_ID, task: str=DEFAULT_TASK, device: int=0, model_kwargs: Optional[dict]=None) -> Any:
    """Inference function to send to the remote hardware.

    Accepts a huggingface model_id and returns a pipeline for the task.
    """
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline
    _model_kwargs = model_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
    try:
        if task == 'text-generation':
            model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
        elif task in ('text2text-generation', 'summarization'):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
        else:
            raise ValueError(f'Got invalid task {task}, currently only {VALID_TASKS} are supported')
    except ImportError as e:
        raise ValueError(f'Could not load the {task} model due to missing dependencies.') from e
    if importlib.util.find_spec('torch') is not None:
        import torch
        cuda_device_count = torch.cuda.device_count()
        if device < -1 or device >= cuda_device_count:
            raise ValueError(f'Got device=={device}, device is required to be within [-1, {cuda_device_count})')
        if device < 0 and cuda_device_count > 0:
            logger.warning('Device has %d GPUs available. Provide device={deviceId} to `from_model_id` to use availableGPUs for execution. deviceId is -1 for CPU and can be a positive integer associated with CUDA device id.', cuda_device_count)
    pipeline = hf_pipeline(task=task, model=model, tokenizer=tokenizer, device=device, model_kwargs=_model_kwargs)
    if pipeline.task not in VALID_TASKS:
        raise ValueError(f'Got invalid task {pipeline.task}, currently only {VALID_TASKS} are supported')
    return pipeline