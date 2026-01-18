from typing import Optional
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
class AsteroidSTFT(nn.Module):

    def __init__(self, fb):
        super(AsteroidSTFT, self).__init__()
        self.enc = Encoder(fb)

    def forward(self, x):
        aux = self.enc(x)
        return to_torchaudio(aux)