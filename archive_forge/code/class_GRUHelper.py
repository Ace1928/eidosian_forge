from typing import Any, Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
class GRUHelper:

    def __init__(self, **params: Any) -> None:
        X = 'X'
        W = 'W'
        R = 'R'
        B = 'B'
        H_0 = 'initial_h'
        LBR = 'linear_before_reset'
        LAYOUT = 'layout'
        number_of_gates = 3
        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f'Missing Required Input: {i}'
        self.num_directions = params[W].shape[0]
        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)
            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]
            layout = params.get(LAYOUT, 0)
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params.get(LBR, 0)
            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.LAYOUT = layout
        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]
        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []
        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))
        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
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