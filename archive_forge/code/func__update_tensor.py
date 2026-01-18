import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _update_tensor(self, tensor_raw):

    def get_tensor_moving_avg(metric_tensor):
        if self.metric_tensor is None:
            self.metric_tensor = np.identity(metric_tensor.shape[0])
        return self.k / (self.k + 1) * self.metric_tensor + 1 / (self.k + 1) * metric_tensor

    def regularize_tensor(metric_tensor):
        tensor_reg = np.real(sqrtm(np.matmul(metric_tensor, metric_tensor)))
        return (tensor_reg + self.reg * np.identity(metric_tensor.shape[0])) / (1 + self.reg)
    tensor_avg = get_tensor_moving_avg(tensor_raw)
    tensor_regularized = regularize_tensor(tensor_avg)
    self.metric_tensor = tensor_regularized
    self.k += 1