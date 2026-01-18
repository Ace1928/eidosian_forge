import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    colorwheel[col:col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    colorwheel[col:col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    colorwheel[col:col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel