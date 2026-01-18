import argparse
import os
import pathlib
import fairseq
import torch
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version
from transformers import (
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging

    Copy/paste/tweak data2vec's weights to our BERT structure.
    