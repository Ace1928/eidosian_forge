import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
def _glue_convert_examples_to_features(examples: List[InputExample], tokenizer: PreTrainedTokenizer, max_length: Optional[int]=None, task=None, label_list=None, output_mode=None):
    if max_length is None:
        max_length = tokenizer.model_max_length
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f'Using label list {label_list} for task {task}')
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info(f'Using output mode {output_mode} for task {task}')
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == 'classification':
            return label_map[example.label]
        elif output_mode == 'regression':
            return float(example.label)
        raise KeyError(output_mode)
    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer([(example.text_a, example.text_b) for example in examples], max_length=max_length, padding='max_length', truncation=True)
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    for i, example in enumerate(examples[:5]):
        logger.info('*** Example ***')
        logger.info(f'guid: {example.guid}')
        logger.info(f'features: {features[i]}')
    return features