from io import BytesIO
import ast
import pickle
import pickletools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import cm
from matplotlib.testing import subprocess_run_helper
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.dates import rrulewrapper
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes  # type: ignore
def _generate_complete_test_figure(fig_ref):
    fig_ref.set_size_inches((10, 6))
    plt.figure(fig_ref)
    plt.suptitle('Can you fit any more in a figure?')
    x, y = (np.arange(8), np.arange(10))
    data = u = v = np.linspace(0, 10, 80).reshape(10, 8)
    v = np.sin(v * -0.6)
    plt.subplot(3, 3, 1)
    plt.plot(list(range(10)))
    plt.ylabel('hello')
    plt.subplot(3, 3, 2)
    plt.contourf(data, hatches=['//', 'ooo'])
    plt.colorbar()
    plt.subplot(3, 3, 3)
    plt.pcolormesh(data)
    plt.subplot(3, 3, 4)
    plt.imshow(data)
    plt.ylabel('hello\nworld!')
    plt.subplot(3, 3, 5)
    plt.pcolor(data)
    ax = plt.subplot(3, 3, 6)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.streamplot(x, y, u, v)
    ax = plt.subplot(3, 3, 7)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.quiver(x, y, u, v)
    plt.subplot(3, 3, 8)
    plt.scatter(x, x ** 2, label='$x^2$')
    plt.legend(loc='upper left')
    plt.subplot(3, 3, 9)
    plt.errorbar(x, x * -0.5, xerr=0.2, yerr=0.4)
    plt.legend(draggable=True)
    fig_ref.align_ylabels()