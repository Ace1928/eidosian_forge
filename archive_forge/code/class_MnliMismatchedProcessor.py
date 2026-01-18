import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format('processor'), FutureWarning)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev_mismatched.tsv')), 'dev_mismatched')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test_mismatched.tsv')), 'test_mismatched')