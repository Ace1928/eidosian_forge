import json
import os
from typing import Optional
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import logging
from ...utils.hub import get_file_from_repo
from ..auto import AutoTokenizer
def _validate_voice_preset_dict(self, voice_preset: Optional[dict]=None):
    for key in ['semantic_prompt', 'coarse_prompt', 'fine_prompt']:
        if key not in voice_preset:
            raise ValueError(f'Voice preset unrecognized, missing {key} as a key.')
        if not isinstance(voice_preset[key], np.ndarray):
            raise ValueError(f'{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.')
        if len(voice_preset[key].shape) != self.preset_shape[key]:
            raise ValueError(f'{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.')