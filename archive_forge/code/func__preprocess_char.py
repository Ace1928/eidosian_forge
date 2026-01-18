import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging
def _preprocess_char(self, text):
    """Special treatment of characters in certain languages"""
    if self.language == 'ron':
        text = text.replace('ț', 'ţ')
    return text