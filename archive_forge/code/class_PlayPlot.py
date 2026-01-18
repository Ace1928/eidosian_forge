from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
class PlayPlot:
    """Provides a callback to create live plots of arbitrary metrics when using :func:`play`.

    This class is instantiated with a function that accepts information about a single environment transition:
        - obs_t: observation before performing action
        - obs_tp1: observation after performing action
        - action: action that was executed
        - rew: reward that was received
        - terminated: whether the environment is terminated or not
        - truncated: whether the environment is truncated or not
        - info: debug info

    It should return a list of metrics that are computed from this data.
    For instance, the function may look like this::

        >>> def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
        ...     return [reward, info["cumulative_reward"], np.linalg.norm(action)]

    :class:`PlayPlot` provides the method :meth:`callback` which will pass its arguments along to that function
    and uses the returned values to update live plots of the metrics.

    Typically, this :meth:`callback` will be used in conjunction with :func:`play` to see how the metrics evolve as you play::

        >>> plotter = PlayPlot(compute_metrics, horizon_timesteps=200,
        ...                    plot_names=["Immediate Rew.", "Cumulative Rew.", "Action Magnitude"])
        >>> play(your_env, callback=plotter.callback)
    """

    def __init__(self, callback: callable, horizon_timesteps: int, plot_names: List[str]):
        """Constructor of :class:`PlayPlot`.

        The function ``callback`` that is passed to this constructor should return
        a list of metrics that is of length ``len(plot_names)``.

        Args:
            callback: Function that computes metrics from environment transitions
            horizon_timesteps: The time horizon used for the live plots
            plot_names: List of plot titles

        Raises:
            DependencyNotInstalled: If matplotlib is not installed
        """
        deprecation('`PlayPlot` is marked as deprecated and will be removed in the near future.')
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names
        if plt is None:
            raise DependencyNotInstalled('matplotlib is not installed, run `pip install gym[other]`')
        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot: List[Optional[plt.Axes]] = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t: ObsType, obs_tp1: ObsType, action: ActType, rew: float, terminated: bool, truncated: bool, info: dict):
        """The callback that calls the provided data callback and adds the data to the plots.

        Args:
            obs_t: The observation at time step t
            obs_tp1: The observation at time step t+1
            action: The action
            rew: The reward
            terminated: If the environment is terminated
            truncated: If the environment is truncated
            info: The information from the environment
        """
        points = self.data_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1
        xmin, xmax = (max(0, self.t - self.horizon_timesteps), self.t)
        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_xlim(xmin, xmax)
        if plt is None:
            raise DependencyNotInstalled('matplotlib is not installed, run `pip install gym[other]`')
        plt.pause(1e-06)