import torch
def _diffsort(a):
    return torch.argsort(torch.diff(a), dim=0, descending=True)