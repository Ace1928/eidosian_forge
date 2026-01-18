import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import IterativeSFTTrainer
def _init_tensor_dummy_dataset(self):
    dummy_dataset_dict = {'input_ids': [torch.tensor([5303, 3621]), torch.tensor([3666, 1438, 318]), torch.tensor([5303, 3621])], 'attention_mask': [torch.tensor([1, 1]), torch.tensor([1, 1, 1]), torch.tensor([1, 1])], 'labels': [torch.tensor([5303, 3621]), torch.tensor([3666, 1438, 318]), torch.tensor([5303, 3621])]}
    dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
    dummy_dataset.set_format('torch')
    return dummy_dataset