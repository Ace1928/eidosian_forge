import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _plot_outliers_with_wave(self, wave, outliers, name):
    import matplotlib
    matplotlib.use(config.get('execution', 'matplotlib_backend'))
    import matplotlib.pyplot as plt
    plt.plot(wave)
    plt.ylim([wave.min(), wave.max()])
    plt.xlim([0, len(wave) - 1])
    if len(outliers):
        plt.plot(np.tile(outliers[:, None], (1, 2)).T, np.tile([wave.min(), wave.max()], (len(outliers), 1)).T, 'r')
    plt.xlabel('Scans - 0-based')
    plt.ylabel(name)