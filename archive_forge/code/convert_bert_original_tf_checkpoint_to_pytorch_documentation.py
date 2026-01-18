import argparse
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging
Convert BERT checkpoint.