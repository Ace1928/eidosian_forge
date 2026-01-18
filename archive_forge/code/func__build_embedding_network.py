import math
import torch
def _build_embedding_network():
    return torch.nn.Sequential(torch.nn.Linear(512 * 4 * 6, 4096), torch.nn.ReLU(True), torch.nn.Linear(4096, 4096), torch.nn.ReLU(True), torch.nn.Linear(4096, 128), torch.nn.ReLU(True))