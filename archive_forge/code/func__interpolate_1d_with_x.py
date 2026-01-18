from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _interpolate_1d_with_x(data: np.ndarray, scale_factor: float, output_width_int: int, x: float, get_coeffs: Callable[[float, float], np.ndarray], roi: np.ndarray | None=None, extrapolation_value: float=0.0, coordinate_transformation_mode: str='half_pixel', exclude_outside: bool=False) -> np.ndarray:
    input_width = len(data)
    output_width = scale_factor * input_width
    if coordinate_transformation_mode == 'align_corners':
        if output_width == 1:
            x_ori = 0.0
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif coordinate_transformation_mode == 'asymmetric':
        x_ori = x / scale_factor
    elif coordinate_transformation_mode == 'tf_crop_and_resize':
        if roi is None:
            raise ValueError('roi cannot be None.')
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori += roi[0] * (input_width - 1)
        if x_ori < 0 or x_ori > input_width - 1:
            return np.array(extrapolation_value)
    elif coordinate_transformation_mode == 'pytorch_half_pixel':
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == 'half_pixel':
        x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == 'half_pixel_symmetric':
        adjustment = output_width_int / output_width
        center = input_width / 2
        offset = center * (1 - adjustment)
        x_ori = offset + (x + 0.5) / scale_factor - 0.5
    else:
        raise ValueError(f'Invalid coordinate_transformation_mode: {coordinate_transformation_mode!r}.')
    x_ori_int = np.floor(x_ori).astype(int).item()
    if x_ori.is_integer():
        ratio = 1
    else:
        ratio = x_ori - x_ori_int
    coeffs = get_coeffs(ratio, scale_factor)
    n = len(coeffs)
    idxes, points = _get_neighbor(x_ori, n, data)
    if exclude_outside:
        for i, idx in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)
    return np.dot(coeffs, points).item()