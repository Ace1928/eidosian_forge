import logging
from ...ndarray import multi_all_finite
from ...ndarray import ndarray as nd
from ... import autograd as ag
class LossScaler(object):
    """Dynamic loss scaler for AMP.

    Properties
    ----------
    loss_scale : float
        The current loss scale
    """

    def __init__(self):
        self._loss_scale = 2.0 ** 16
        self._next_loss_scale = self._loss_scale
        self._max_loss_scale = 2.0 ** 24
        self._scale_seq_len = 2000
        self._unskipped = 0

    @property
    def loss_scale(self):
        return self._loss_scale

    def has_overflow(self, params):
        """Check gradients for overflow."""
        with ag.pause():
            chunk_size = 200
            valid_params = [p._grad[0] for p in params if p._grad is not None]
            gpu_output = nd.ones((1,), ctx=valid_params[0].context)
            nb_params = len(valid_params)
            for idx in range(0, nb_params, chunk_size):
                multi_all_finite(*valid_params[idx:idx + chunk_size], num_arrays=len(valid_params[idx:idx + chunk_size]), init_output=False, out=gpu_output)
        has_overflow = not bool(gpu_output.asnumpy())
        self._loss_scale = self._next_loss_scale
        if has_overflow:
            self._next_loss_scale = self._loss_scale / 2.0
            self._unskipped = 0
            logging.info('AMP: decreasing loss scale to %f', self._next_loss_scale)
        else:
            self._unskipped += 1
        if self._unskipped == self._scale_seq_len:
            self._unskipped = 0
            self._next_loss_scale = min(self._max_loss_scale, self._loss_scale * 2.0)
            logging.info('AMP: increasing loss scale to %f', self._next_loss_scale)
        return has_overflow