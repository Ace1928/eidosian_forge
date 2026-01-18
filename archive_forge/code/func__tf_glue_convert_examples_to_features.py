import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
def _tf_glue_convert_examples_to_features(examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int]=None) -> tf.data.Dataset:
    """
        Returns:
            A `tf.data.Dataset` containing the task-specific features.

        """
    processor = glue_processors[task]()
    examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
    features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    label_type = tf.float32 if task == 'sts-b' else tf.int64

    def gen():
        for ex in features:
            d = {k: v for k, v in asdict(ex).items() if v is not None}
            label = d.pop('label')
            yield (d, label)
    input_names = tokenizer.model_input_names
    return tf.data.Dataset.from_generator(gen, ({k: tf.int32 for k in input_names}, label_type), ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])))