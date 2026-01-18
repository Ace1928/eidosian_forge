import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from ...models.bert.tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from ...utils import is_tf_available, is_torch_available, logging
from .utils import DataProcessor
def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
    if not evaluate:
        answer = tensor_dict['answers']['text'][0].numpy().decode('utf-8')
        answer_start = tensor_dict['answers']['answer_start'][0].numpy()
        answers = []
    else:
        answers = [{'answer_start': start.numpy(), 'text': text.numpy().decode('utf-8')} for start, text in zip(tensor_dict['answers']['answer_start'], tensor_dict['answers']['text'])]
        answer = None
        answer_start = None
    return SquadExample(qas_id=tensor_dict['id'].numpy().decode('utf-8'), question_text=tensor_dict['question'].numpy().decode('utf-8'), context_text=tensor_dict['context'].numpy().decode('utf-8'), answer_text=answer, start_position_character=answer_start, title=tensor_dict['title'].numpy().decode('utf-8'), answers=answers)