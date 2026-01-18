from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _heatmap(self, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
    """
        Draw 2D heatmaps for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 2D heatmap, it should be the second design variable in the design_ranges
        ylabel_text: y label title, a string.
            In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 2D heatmap for each criteria
        """
    sensitivity_dict = {}
    for i, name in enumerate(self.design_names):
        if name in self.sensitivity_dimension:
            sensitivity_dict[name] = self.design_ranges[i]
        elif name[0] in self.sensitivity_dimension:
            sensitivity_dict[name[0]] = self.design_ranges[i]
    x_range = sensitivity_dict[self.sensitivity_dimension[0]]
    y_range = sensitivity_dict[self.sensitivity_dimension[1]]
    A_range = self.figure_result_data['A'].values.tolist()
    D_range = self.figure_result_data['D'].values.tolist()
    E_range = self.figure_result_data['E'].values.tolist()
    ME_range = self.figure_result_data['ME'].values.tolist()
    cri_a = np.asarray(A_range).reshape(len(x_range), len(y_range))
    cri_d = np.asarray(D_range).reshape(len(x_range), len(y_range))
    cri_e = np.asarray(E_range).reshape(len(x_range), len(y_range))
    cri_e_cond = np.asarray(ME_range).reshape(len(x_range), len(y_range))
    self.cri_a = cri_a
    self.cri_d = cri_d
    self.cri_e = cri_e
    self.cri_e_cond = cri_e_cond
    if log_scale:
        hes_a = np.log10(self.cri_a)
        hes_e = np.log10(self.cri_e)
        hes_d = np.log10(self.cri_d)
        hes_e2 = np.log10(self.cri_e_cond)
    else:
        hes_a = self.cri_a
        hes_e = self.cri_e
        hes_d = self.cri_d
        hes_e2 = self.cri_e_cond
    xLabel = x_range
    yLabel = y_range
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    plt.pyplot.rcParams.update(params)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_ylabel(ylabel_text)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel(xlabel_text)
    im = ax.imshow(hes_a.T, cmap=plt.pyplot.cm.hot_r)
    ba = plt.pyplot.colorbar(im)
    ba.set_label('log10(trace(FIM))')
    plt.pyplot.title(title_text + ' - A optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    plt.pyplot.rcParams.update(params)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_ylabel(ylabel_text)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel(xlabel_text)
    im = ax.imshow(hes_d.T, cmap=plt.pyplot.cm.hot_r)
    ba = plt.pyplot.colorbar(im)
    ba.set_label('log10(det(FIM))')
    plt.pyplot.title(title_text + ' - D optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    plt.pyplot.rcParams.update(params)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_ylabel(ylabel_text)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel(xlabel_text)
    im = ax.imshow(hes_e.T, cmap=plt.pyplot.cm.hot_r)
    ba = plt.pyplot.colorbar(im)
    ba.set_label('log10(minimal eig(FIM))')
    plt.pyplot.title(title_text + ' - E optimality')
    plt.pyplot.show()
    fig = plt.pyplot.figure()
    plt.pyplot.rc('axes', titlesize=font_axes)
    plt.pyplot.rc('axes', labelsize=font_axes)
    plt.pyplot.rc('xtick', labelsize=font_tick)
    plt.pyplot.rc('ytick', labelsize=font_tick)
    ax = fig.add_subplot(111)
    params = {'mathtext.default': 'regular'}
    plt.pyplot.rcParams.update(params)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_ylabel(ylabel_text)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel(xlabel_text)
    im = ax.imshow(hes_e2.T, cmap=plt.pyplot.cm.hot_r)
    ba = plt.pyplot.colorbar(im)
    ba.set_label('log10(cond(FIM))')
    plt.pyplot.title(title_text + ' - Modified E-optimality')
    plt.pyplot.show()