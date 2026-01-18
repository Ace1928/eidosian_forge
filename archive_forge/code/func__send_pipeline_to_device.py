import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.utils import enforce_stop_tokens
def _send_pipeline_to_device(pipeline: Any, device: int) -> Any:
    """Send a pipeline to a device on the cluster."""
    if isinstance(pipeline, str):
        with open(pipeline, 'rb') as f:
            pipeline = pickle.load(f)
    if importlib.util.find_spec('torch') is not None:
        import torch
        cuda_device_count = torch.cuda.device_count()
        if device < -1 or device >= cuda_device_count:
            raise ValueError(f'Got device=={device}, device is required to be within [-1, {cuda_device_count})')
        if device < 0 and cuda_device_count > 0:
            logger.warning('Device has %d GPUs available. Provide device={deviceId} to `from_model_id` to use availableGPUs for execution. deviceId is -1 for CPU and can be a positive integer associated with CUDA device id.', cuda_device_count)
        pipeline.device = torch.device(device)
        pipeline.model = pipeline.model.to(pipeline.device)
    return pipeline