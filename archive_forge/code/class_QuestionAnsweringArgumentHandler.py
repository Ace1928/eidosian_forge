import inspect
import types
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ..data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
    internal [`SquadExample`].

    QuestionAnsweringArgumentHandler manages all the possible to create a [`SquadExample`] from the command-line
    supplied arguments.
    """

    def normalize(self, item):
        if isinstance(item, SquadExample):
            return item
        elif isinstance(item, dict):
            for k in ['question', 'context']:
                if k not in item:
                    raise KeyError('You need to provide a dictionary with keys {question:..., context:...}')
                elif item[k] is None:
                    raise ValueError(f'`{k}` cannot be None')
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError(f'`{k}` cannot be empty')
            return QuestionAnsweringPipeline.create_sample(**item)
        raise ValueError(f'{item} argument needs to be of type (SquadExample, dict)')

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            if len(args) == 1:
                inputs = args[0]
            elif len(args) == 2 and {type(el) for el in args} == {str}:
                inputs = [{'question': args[0], 'context': args[1]}]
            else:
                inputs = list(args)
        elif 'X' in kwargs:
            inputs = kwargs['X']
        elif 'data' in kwargs:
            inputs = kwargs['data']
        elif 'question' in kwargs and 'context' in kwargs:
            if isinstance(kwargs['question'], list) and isinstance(kwargs['context'], str):
                inputs = [{'question': Q, 'context': kwargs['context']} for Q in kwargs['question']]
            elif isinstance(kwargs['question'], list) and isinstance(kwargs['context'], list):
                if len(kwargs['question']) != len(kwargs['context']):
                    raise ValueError("Questions and contexts don't have the same lengths")
                inputs = [{'question': Q, 'context': C} for Q, C in zip(kwargs['question'], kwargs['context'])]
            elif isinstance(kwargs['question'], str) and isinstance(kwargs['context'], str):
                inputs = [{'question': kwargs['question'], 'context': kwargs['context']}]
            else:
                raise ValueError("Arguments can't be understood")
        else:
            raise ValueError(f'Unknown arguments {kwargs}')
        generator_types = (types.GeneratorType, Dataset) if Dataset is not None else (types.GeneratorType,)
        if isinstance(inputs, generator_types):
            return inputs
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, Iterable):
            inputs = list(inputs)
        else:
            raise ValueError(f'Invalid arguments {kwargs}')
        for i, item in enumerate(inputs):
            inputs[i] = self.normalize(item)
        return inputs