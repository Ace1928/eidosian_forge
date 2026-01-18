import torch.nn as nn
from fairscale.optim import GradScaler
class Offload_Sequential:

    def get_model_config():
        return {'inputs': 100, 'outputs': 5, 'hidden': 1000, 'layers': 100, 'clip_value': 0.05}

    def get_benchmark_config():
        return {'epochs': 1, 'lr': 0.001, 'batch_size': 8, 'criterion': nn.CrossEntropyLoss(), 'slices': 3, 'checkpoint_activation': True, 'num_microbatches': 1}