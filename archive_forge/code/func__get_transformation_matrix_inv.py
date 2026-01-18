from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@classmethod
def _get_transformation_matrix_inv(cls, saxis):
    saxis = saxis / np.linalg.norm(saxis)
    alpha = np.arctan2(saxis[1], saxis[0])
    beta = np.arctan2(np.sqrt(saxis[0] ** 2 + saxis[1] ** 2), saxis[2])
    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)
    sin_a = np.sin(alpha)
    sin_b = np.sin(beta)
    return [[cos_b * cos_a, cos_b * sin_a, -sin_b], [-sin_a, cos_a, 0], [sin_b * cos_a, sin_b * sin_a, cos_b]]