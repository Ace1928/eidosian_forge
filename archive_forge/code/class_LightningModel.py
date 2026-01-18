import argparse
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import LongformerForQuestionAnswering, LongformerModel
class LightningModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_labels = 2
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.num_labels)

    def forward(self):
        pass