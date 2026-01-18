import argparse
import torch
from ...utils import logging
from . import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert
Convert ALBERT checkpoint.