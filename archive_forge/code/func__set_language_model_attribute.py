import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends
@staticmethod
def _set_language_model_attribute(decoder: 'BeamSearchDecoderCTC', attribute: str, value: float):
    setattr(decoder.model_container[decoder._model_key], attribute, value)