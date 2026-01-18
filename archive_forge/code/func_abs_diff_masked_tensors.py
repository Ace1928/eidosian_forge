import copy
import fnmatch
import gc
import re
import tempfile
import unittest
import pytest
import torch
from huggingface_hub import HfApi, HfFolder, delete_repo
from parameterized import parameterized
from pytest import mark
from requests.exceptions import HTTPError
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import respond_to_batch
from .testing_constants import CI_HUB_ENDPOINT, CI_HUB_USER, CI_HUB_USER_TOKEN
from .testing_utils import require_peft, require_torch_multi_gpu
def abs_diff_masked_tensors(tensor_1, tensor_2, mask_1, mask_2):
    diffs = []
    for l1, l2, m1, m2 in zip(tensor_1, tensor_2, mask_1, mask_2):
        diff = apply_mask(l1, m1) - apply_mask(l2, m2)
        diffs.append(diff.sum())
    return abs(sum(diffs))