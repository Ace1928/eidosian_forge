import warnings
from ..trainer import Trainer
from ..utils import logging
class SageMakerTrainer(Trainer):

    def __init__(self, args=None, **kwargs):
        warnings.warn('`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` instead.', FutureWarning)
        super().__init__(args=args, **kwargs)