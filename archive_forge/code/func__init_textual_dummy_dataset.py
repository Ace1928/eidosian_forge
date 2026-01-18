import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import IterativeSFTTrainer
def _init_textual_dummy_dataset(self):
    dummy_dataset_dict = {'texts': ['Testing the IterativeSFTTrainer.', 'This is a test of the IterativeSFTTrainer'], 'texts_labels': ['Testing the IterativeSFTTrainer.', 'This is a test of the IterativeSFTTrainer']}
    dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
    dummy_dataset.set_format('torch')
    return dummy_dataset