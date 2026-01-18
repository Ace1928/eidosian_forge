import torch.nn as nn
from fairscale.optim import GradScaler
class MOE:

    def get_model_config():
        return {'vocab_size': 10000, 'ninp': 1024, 'nhid': 4096, 'nhead': 32, 'dropout': 0, 'initrange': 0.1, 'scaler': GradScaler(), 'clip_value': 0.05, 'num_decoder_layers': 20, 'seq_len': 33, 'is_moe': True, 'num_local_experts': 2}

    def get_benchmark_config():
        return {'epochs': 1, 'lr': 0.001, 'batch_size': 32, 'criterion': nn.CrossEntropyLoss()}