import argparse
import os
import pickle as pkl
import torch
from torch import nn
from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM
class OriginalMegaForMaskedLM(nn.Module):
    """A wrapper class for doing masked language modeling with Mega"""

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        self.mega = MegaLM(mega_args, depth, vocab_size)
        self.mlm_head = nn.Linear(mega_args.encoder_embed_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Perform a forward pass through the Mega encoder and the masked LM head. Returns logits for each vocabulary
        entry.

        If `batch_first` (default to align with Hugging Face tokenizer behavior), output will have the shape (Batch
        size, Sequence length, Vocab size); otherwise (S, B, V)
        """
        encoder_output = self.mega(input_ids, attention_mask, batch_first, ignore_mask_value)
        return self.mlm_head(self.dropout(encoder_output))