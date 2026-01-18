import argparse
import os
import gluonnlp as nlp
import mxnet as mx
import numpy as np
import torch
from gluonnlp.base import get_home_dir
from gluonnlp.model.bert import BERTEncoder
from gluonnlp.model.utils import _load_vocab
from gluonnlp.vocab import Vocab
from packaging import version
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel, RobertaTokenizer
from transformers.models.bert.modeling_bert import (
from transformers.utils import logging

    Convert the original Bort checkpoint (based on MXNET and Gluonnlp) to our BERT structure-
    