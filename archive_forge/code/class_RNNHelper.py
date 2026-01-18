from typing import Any, Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
class RNNHelper:

    def __init__(self, **params: Any) -> None:
        X = 'X'
        W = 'W'
        R = 'R'
        B = 'B'
        H_0 = 'initial_h'
        LAYOUT = 'layout'
        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f'Missing Required Input: {i}'
        self.num_directions = params[str(W)].shape[0]
        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)
            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]
            layout = params.get(LAYOUT, 0)
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = params[B] if B in params else np.zeros(2 * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)
            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LAYOUT = layout
        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]
        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []
        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            H = self.f(np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2)))
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated
        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]
        return (Y, Y_h)