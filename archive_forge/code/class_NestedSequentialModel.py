from typing import Tuple
import torch
import torch.nn as nn
class NestedSequentialModel(nn.Module):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.seq1 = nn.Sequential(nn.Linear(1, 1, device=device), FakeSequential(nn.Linear(1, 1, device=device), nn.ReLU(), FakeSequential(nn.Linear(1, 1, device=device)), nn.ReLU()), nn.Linear(1, 2, device=device))
        self.lin = nn.Linear(2, 2, device=device)
        self.seq2 = nn.Sequential(nn.ReLU(), nn.Linear(2, 3, device=device), FakeSequential(nn.Linear(3, 2, bias=False, device=device), nn.Linear(2, 4, bias=False, device=device)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.seq2(self.lin(self.seq1(x)))