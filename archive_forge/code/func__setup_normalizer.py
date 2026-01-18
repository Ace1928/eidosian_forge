import json
import os
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def _setup_normalizer(self):
    try:
        from sacremoses import MosesPunctNormalizer
        self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
    except (ImportError, FileNotFoundError):
        warnings.warn('Recommended: pip install sacremoses.')
        self.punc_normalizer = lambda x: x