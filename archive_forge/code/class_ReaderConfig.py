from enum import Enum
from typing import Any, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.llms.utils import enforce_stop_tokens
class ReaderConfig(BaseModel):

    class Config:
        protected_namespaces = ()
    model_name: str
    'The name of the model to use'
    device: Device = Device.cuda
    'The device to use for inference, cuda or cpu'
    consumer_group: str = 'primary'
    'The consumer group to place the reader into'
    tensor_parallel: Optional[int] = None
    'The number of gpus you would like your model to be split across'
    max_seq_length: int = 512
    'The maximum sequence length to use for inference, defaults to 512'
    max_batch_size: int = 4
    'The max batch size for continuous batching of requests'