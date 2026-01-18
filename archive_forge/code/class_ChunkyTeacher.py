from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
class ChunkyTeacher(ChunkTeacher):

    def _get_data_folder(self):
        return None

    def get_num_samples(self, opt) -> Tuple[int, int]:
        datatype = opt['datatype']
        if 'train' in datatype:
            return (NUM_TRAIN, NUM_TRAIN)
        elif 'valid' in datatype:
            return (NUM_TEST, NUM_TEST)
        elif 'test' in datatype:
            return (NUM_TEST, NUM_TEST)

    def get_fold_chunks(self, opt) -> List[int]:
        datatype = opt['datatype']
        if 'train' in datatype:
            return list(range(50))
        elif 'valid' in datatype:
            return list(range(50, 60))
        elif 'test' in datatype:
            return list(range(60, 70))

    def load_from_chunk(self, chunk_idx: int):
        output = []
        for i in range(10):
            text = ' '.join([str(i)] + [str(chunk_idx)] * 5)
            resp = ' '.join([str(i)])
            output.append((text, resp))
        return output

    def create_message(self, sample_item, entry_idx=0):
        text, label = sample_item
        return {'text': text, 'labels': [label], 'episode_done': True}