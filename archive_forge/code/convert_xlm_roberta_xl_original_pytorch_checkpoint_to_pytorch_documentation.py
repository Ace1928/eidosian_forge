import argparse
import pathlib
import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version
from transformers import XLMRobertaConfig, XLMRobertaXLForMaskedLM, XLMRobertaXLForSequenceClassification
from transformers.models.bert.modeling_bert import (
from transformers.models.roberta.modeling_roberta import RobertaAttention
from transformers.utils import logging

    Copy/paste/tweak roberta's weights to our BERT structure.
    