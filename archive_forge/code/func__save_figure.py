import os
import subprocess
import sys
import pytest
import matplotlib as mpl
import matplotlib.testing.compare
from matplotlib import pyplot as plt
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
def _save_figure(objects='mhi', fmt='pdf', usetex=False):
    mpl.use(fmt)
    mpl.rcParams.update({'svg.hashsalt': 'asdf', 'text.usetex': usetex})
    fig = plt.figure()
    if 'm' in objects:
        ax1 = fig.add_subplot(1, 6, 1)
        x = range(10)
        ax1.plot(x, [1] * 10, marker='D')
        ax1.plot(x, [2] * 10, marker='x')
        ax1.plot(x, [3] * 10, marker='^')
        ax1.plot(x, [4] * 10, marker='H')
        ax1.plot(x, [5] * 10, marker='v')
    if 'h' in objects:
        ax2 = fig.add_subplot(1, 6, 2)
        bars = ax2.bar(range(1, 5), range(1, 5)) + ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5))
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])
        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)
    if 'i' in objects:
        A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
        A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
        fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
        A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
        fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')
    x = range(5)
    ax = fig.add_subplot(1, 6, 6)
    ax.plot(x, x)
    ax.set_title('A string $1+2+\\sigma$')
    ax.set_xlabel('A string $1+2+\\sigma$')
    ax.set_ylabel('A string $1+2+\\sigma$')
    stdout = getattr(sys.stdout, 'buffer', sys.stdout)
    fig.savefig(stdout, format=fmt)