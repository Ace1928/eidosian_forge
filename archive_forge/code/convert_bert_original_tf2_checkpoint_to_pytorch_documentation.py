import argparse
import os
import re
import tensorflow as tf
import torch
from transformers import BertConfig, BertModel
from transformers.utils import logging

This script can be used to convert a head-less TF2.x Bert model to PyTorch, as published on the official (now
deprecated) GitHub: https://github.com/tensorflow/models/tree/v2.3.0/official/nlp/bert

TF2.x uses different variable names from the original BERT (TF 1.4) implementation. The script re-maps the TF2.x Bert
weight names to the original names, so the model can be imported with Huggingface/transformer.

You may adapt this script to include classification/MLM/NSP/etc. heads.

Note: This script is only working with an older version of the TensorFlow models repository (<= v2.3.0).
      Models trained with never versions are not compatible with this script.
