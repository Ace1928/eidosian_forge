import inspect
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
import huggingface_hub
import numpy as np
from packaging import version
from .. import (
from ..utils import TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging
from . import BaseTransformersCLICommand
def _get_audio_input():
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    speech_samples = ds.sort('id').select(range(2))[:2]['audio']
    raw_samples = [x['array'] for x in speech_samples]
    return raw_samples