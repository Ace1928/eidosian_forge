import argparse
import torch
from transformers import RemBertConfig, RemBertModel, load_tf_weights_in_rembert
from transformers.utils import logging
Convert RemBERT checkpoint.