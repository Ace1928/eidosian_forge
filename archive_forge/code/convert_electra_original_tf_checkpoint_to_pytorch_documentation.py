import argparse
import torch
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from transformers.utils import logging
Convert ELECTRA checkpoint.