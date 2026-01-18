import json
import shlex
import subprocess
from typing import Tuple
import torch
class OutlierTracer:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self, model):
        self.last_w = None
        self.current_outlier_dims = None
        self.hvalues = []
        self.outliers = []
        self.hvalue2outlier_idx = {}
        self.initialized = True
        self.hooks = []
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(outlier_hook))

    def is_initialized(self):
        return getattr(self, 'initialized', False)

    def get_hvalue(self, weight):
        return weight.data.storage().data_ptr()

    def get_outliers(self, weight):
        if not self.is_initialized():
            print('Outlier tracer is not initialized...')
            return None
        hvalue = self.get_hvalue(weight)
        if hvalue in self.hvalue2outlier_idx:
            return self.hvalue2outlier_idx[hvalue]
        else:
            return None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance