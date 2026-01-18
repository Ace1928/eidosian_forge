import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
    subres = {}
    for ln in [x.strip() for x in lns]:
        if not ln.startswith('*'):
            continue
        ln = ln[1:].strip()
        for k in ['download', 'dataset', 'models', 'model', 'pre-processing']:
            if ln.startswith(k):
                break
        else:
            continue
        if k in ['dataset', 'model', 'pre-processing']:
            splat = ln.split(':')
            _, v = splat
            subres[k] = v
        elif k == 'download':
            v = ln.split('(')[-1][:-1]
            subres[k] = v
    return subres