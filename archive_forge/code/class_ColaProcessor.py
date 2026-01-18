import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format('processor'), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(), tensor_dict['sentence'].numpy().decode('utf-8'), None, str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test.tsv')), 'test')

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == 'test'
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for i, line in enumerate(lines):
            guid = f'{set_type}-{i}'
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples