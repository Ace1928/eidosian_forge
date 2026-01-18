import argparse
import os
from functools import reduce
import fairseq
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, logging
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig
from transformers.models.data2vec.data2vec_audio import Data2VecAudioModel as Dummy  # noqa: F401
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioForCTC, Data2VecAudioModel

    Copy/paste/tweak model's weights to transformers design.
    