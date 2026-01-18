import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
class TomographyResult:
    """Results from a state tomography experiment."""

    def __init__(self, density_matrix: np.ndarray):
        """Inits TomographyResult.

        Args:
            density_matrix: The density matrix obtained from tomography.
        """
        self._density_matrix = density_matrix

    @property
    def data(self) -> np.ndarray:
        """Returns an n^2 by n^2 complex matrix representing the density
        matrix of the n-qubit system.
        """
        return self._density_matrix

    def plot(self, axes: Optional[List[plt.Axes]]=None, **plot_kwargs: Any) -> List[plt.Axes]:
        """Plots the real and imaginary parts of the density matrix as two 3D bar plots.

        Args:
            axes: A list of 2 `plt.Axes` instances. Note that they must be in
                3d projections. If not given, a new figure is created with 2
                axes and the plotted figure is shown.
            **plot_kwargs: The optional kwargs passed to bar3d.

        Returns:
            the list of `plt.Axes` being plotted on.

        Raises:
            ValueError: If axes is a list with length != 2.
        """
        show_plot = axes is None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), subplot_kw={'projection': '3d'})
        elif len(axes) != 2:
            raise ValueError('A TomographyResult needs 2 axes to plot.')
        mat = self._density_matrix
        a, _ = mat.shape
        num_qubits = int(np.log2(a))
        state_labels = [[0, 1]] * num_qubits
        kets = []
        for label in itertools.product(*state_labels):
            kets.append('|' + str(list(label))[1:-1] + '>')
        mat_re = np.real(mat)
        mat_im = np.imag(mat)
        _matrix_bar_plot(mat_re, 'Real($\\rho$)', axes[0], kets, 'Density Matrix (Real Part)', ylim=(-1, 1), **plot_kwargs)
        _matrix_bar_plot(mat_im, 'Imaginary($\\rho$)', axes[1], kets, 'Density Matrix (Imaginary Part)', ylim=(-1, 1), **plot_kwargs)
        if show_plot:
            fig.show()
        return axes