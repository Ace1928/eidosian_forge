import argparse
from pathlib import Path
import fairseq
import torch
from fairseq.models.xmod import XMODModel as FairseqXmodModel
from packaging import version
from transformers import XmodConfig, XmodForMaskedLM, XmodForSequenceClassification
from transformers.utils import logging
Convert X-MOD checkpoint.