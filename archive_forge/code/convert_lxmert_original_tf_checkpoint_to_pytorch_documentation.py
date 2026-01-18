import argparse
import torch
from transformers import LxmertConfig, LxmertForPreTraining, load_tf_weights_in_lxmert
from transformers.utils import logging
Convert LXMERT checkpoint.