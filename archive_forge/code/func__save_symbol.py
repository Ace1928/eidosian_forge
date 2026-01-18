import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def _save_symbol(self, estimator):
    symbol_file = os.path.join(self.model_dir, self.model_prefix + '-symbol.json')
    if hasattr(estimator.net, '_cached_graph') and estimator.net._cached_graph:
        sym = estimator.net._cached_graph[1]
        sym.save(symbol_file)
    else:
        estimator.logger.info('Model architecture(symbol file) is not saved, please use HybridBlock to construct your model, and call net.hybridize() before passing to Estimator in order to save model architecture as %s.', symbol_file)