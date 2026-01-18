import argparse
import os
import pickle as pkl
import torch
from torch import nn
from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM
class MegaLM(nn.Module):
    """The base class for our Mega encoder - given input IDs, embed text and return encoder output"""

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        self.mega_args = mega_args
        self.embedding_layer = nn.Embedding(vocab_size, self.mega_args.encoder_embed_dim)
        self.encoders = nn.ModuleList([MegaEncoderLayer(self.mega_args) for _ in range(depth)])
        self.depth = depth

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Code for a forward pass - expects input_ids and attention_mask to come from a Hugging Face tokenizer as PyTorch
        tensors, and returns a tensor of size (batch, n_classes) containing classification logits

        Other options:
          - batch_first: boolean indicating whether the batch dimension is first in input_ids (default: True, which
            aligns with the HF tokenizer behavior)
          - ignore_mask_value: the value in attention_mask that identifies tokens that should be ignored (default: 0,
            which aligns with HF tokenizer)
        """
        if batch_first:
            input_ids = input_ids.T
        if ignore_mask_value == 0:
            attention_mask = 1 - attention_mask
        embeds = self.embedding_layer(input_ids)
        for encoder in self.encoders:
            embeds = encoder(embeds, attention_mask)
        if batch_first:
            return torch.transpose(embeds, 0, 1)
        else:
            return embeds