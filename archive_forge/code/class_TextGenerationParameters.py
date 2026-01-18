import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class TextGenerationParameters:
    """
    Parameters for text generation.

    Args:
        do_sample (`bool`, *optional*):
            Activate logits sampling. Defaults to False.
        max_new_tokens (`int`, *optional*):
            Maximum number of generated tokens. Defaults to 20.
        repetition_penalty (`Optional[float]`, *optional*):
            The parameter for repetition penalty. A value of 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf)
            for more details. Defaults to None.
        return_full_text (`bool`, *optional*):
            Whether to prepend the prompt to the generated text. Defaults to False.
        stop (`List[str]`, *optional*):
            Stop generating tokens if a member of `stop_sequences` is generated. Defaults to an empty list.
        seed (`Optional[int]`, *optional*):
            Random sampling seed. Defaults to None.
        temperature (`Optional[float]`, *optional*):
            The value used to modulate the logits distribution. Defaults to None.
        top_k (`Optional[int]`, *optional*):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (`Optional[float]`, *optional*):
            If set to a value less than 1, only the smallest set of most probable tokens with probabilities that add up
            to `top_p` or higher are kept for generation. Defaults to None.
        truncate (`Optional[int]`, *optional*):
            Truncate input tokens to the given size. Defaults to None.
        typical_p (`Optional[float]`, *optional*):
            Typical Decoding mass. See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)
            for more information. Defaults to None.
        best_of (`Optional[int]`, *optional*):
            Generate `best_of` sequences and return the one with the highest token logprobs. Defaults to None.
        watermark (`bool`, *optional*):
            Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226). Defaults to False.
        details (`bool`, *optional*):
            Get generation details. Defaults to False.
        decoder_input_details (`bool`, *optional*):
            Get decoder input token logprobs and ids. Defaults to False.
    """
    do_sample: bool = False
    max_new_tokens: int = 20
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    stop: List[str] = field(default_factory=lambda: [])
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    best_of: Optional[int] = None
    watermark: bool = False
    details: bool = False
    decoder_input_details: bool = False

    @validator('best_of')
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValueError('`best_of` must be strictly positive')
            if field_value > 1 and values['seed'] is not None:
                raise ValueError('`seed` must not be set when `best_of` is > 1')
            sampling = values['do_sample'] | (values['temperature'] is not None) | (values['top_k'] is not None) | (values['top_p'] is not None) | (values['typical_p'] is not None)
            if field_value > 1 and (not sampling):
                raise ValueError('you must use sampling when `best_of` is > 1')
        return field_value

    @validator('repetition_penalty')
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError('`repetition_penalty` must be strictly positive')
        return v

    @validator('seed')
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError('`seed` must be positive')
        return v

    @validator('temperature')
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValueError('`temperature` must be strictly positive')
        return v

    @validator('top_k')
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError('`top_k` must be strictly positive')
        return v

    @validator('top_p')
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError('`top_p` must be > 0.0 and < 1.0')
        return v

    @validator('truncate')
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValueError('`truncate` must be strictly positive')
        return v

    @validator('typical_p')
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError('`typical_p` must be > 0.0 and < 1.0')
        return v