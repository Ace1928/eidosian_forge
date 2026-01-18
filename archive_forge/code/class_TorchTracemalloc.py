import argparse
import gc
import json
import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator, DistributedType
from accelerate.utils import is_npu_available, is_xpu_available
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
class TorchTracemalloc:

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            self.begin = torch.cuda.memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            torch.npu.reset_max_memory_allocated()
            self.begin = torch.npu.memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()
            self.begin = torch.xpu.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            self.end = torch.npu.memory_allocated()
            self.peak = torch.npu.max_memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            self.end = torch.xpu.memory_allocated()
            self.peak = torch.xpu.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)