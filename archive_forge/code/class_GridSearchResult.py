from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
class GridSearchResult:

    def __init__(self, design_ranges, design_dimension_names, FIM_result_list, store_optimality_name=None):
        """
        This class deals with the FIM results from grid search, providing A, D, E, ME-criteria results for each design variable.
        Can choose to draw 1D sensitivity curves and 2D heatmaps.

        Parameters
        ----------
        design_ranges:
            a ``dict`` whose keys are design variable names, values are a list of design variable values to go over
        design_dimension_names:
            a ``list`` of design variables names
        FIM_result_list:
            a ``dict`` containing FIM results, keys are a tuple of design variable values, values are FIM result objects
        store_optimality_name:
            a .csv file name containing all four optimalities value
        """
        self.design_names = design_dimension_names
        self.design_ranges = design_ranges
        self.FIM_result_list = FIM_result_list
        self.store_optimality_name = store_optimality_name

    def extract_criteria(self):
        """
        Extract design criteria values for every 'grid' (design variable combination) searched.

        Returns
        -------
        self.store_all_results_dataframe: a pandas dataframe with columns as design variable names and A, D, E, ME-criteria names.
            Each row contains the design variable value for this 'grid', and the 4 design criteria value for this 'grid'.
        """
        store_all_results = []
        search_design_set = product(*self.design_ranges)
        for design_set_iter in search_design_set:
            result_object_asdict = {k: v for k, v in self.FIM_result_list.items() if k == design_set_iter}
            result_object_iter = result_object_asdict[design_set_iter]
            store_iteration_result = list(design_set_iter)
            store_iteration_result.append(result_object_iter.trace)
            store_iteration_result.append(result_object_iter.det)
            store_iteration_result.append(result_object_iter.min_eig)
            store_iteration_result.append(result_object_iter.cond)
            store_all_results.append(store_iteration_result)
        column_names = []
        for i in self.design_names:
            if type(i) is list:
                column_names.append(i[0])
            else:
                column_names.append(i)
        column_names.append('A')
        column_names.append('D')
        column_names.append('E')
        column_names.append('ME')
        store_all_results = np.asarray(store_all_results)
        self.store_all_results_dataframe = pd.DataFrame(store_all_results, columns=column_names)
        if self.store_optimality_name is not None:
            self.store_all_results_dataframe.to_csv(self.store_optimality_name, index=False)

    def figure_drawing(self, fixed_design_dimensions, sensitivity_dimension, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
        """
        Extract results needed for drawing figures from the overall result dataframe.
        Draw 1D sensitivity curve or 2D heatmap.
        It can be applied to results of any dimensions, but requires design variable values in other dimensions be fixed.

        Parameters
        ----------
        fixed_design_dimensions: a dictionary, keys are the design variable names to be fixed, values are the value of it to be fixed.
        sensitivity_dimension: a list of design variable names to draw figures.
            If only one name is given, a 1D sensitivity curve is drawn
            if two names are given, a 2D heatmap is drawn.
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
            In a 2D heatmap, it should be the second design variable in the design_ranges
        ylabel_text: y label title, a string.
            A 1D sensitivity curve does not need it. In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        None
        """
        self.fixed_design_names = list(fixed_design_dimensions.keys())
        self.fixed_design_values = list(fixed_design_dimensions.values())
        self.sensitivity_dimension = sensitivity_dimension
        if len(self.fixed_design_names) + len(self.sensitivity_dimension) != len(self.design_names):
            raise ValueError('Error: All dimensions except for those the figures are drawn by should be fixed.')
        if len(self.sensitivity_dimension) not in [1, 2]:
            raise ValueError('Error: Either 1D or 2D figures can be drawn.')
        if len(self.fixed_design_names) != 0:
            filter = ''
            for i in range(len(self.fixed_design_names)):
                filter += '(self.store_all_results_dataframe['
                filter += str(self.fixed_design_names[i])
                filter += ']=='
                filter += str(self.fixed_design_values[i])
                filter += ')'
                if i != len(self.fixed_design_names) - 1:
                    filter += '&'
            figure_result_data = self.store_all_results_dataframe.loc[eval(filter)]
        else:
            figure_result_data = self.store_all_results_dataframe
        self.figure_result_data = figure_result_data
        if len(sensitivity_dimension) == 1:
            self._curve1D(title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True)
        elif len(sensitivity_dimension) == 2:
            self._heatmap(title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True)

    def _curve1D(self, title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True):
        """
        Draw 1D sensitivity curves for all design criteria

        Parameters
        ----------
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        4 Figures of 1D sensitivity curves for each criteria
        """
        x_range = self.figure_result_data[self.sensitivity_dimension[0]].values.tolist()
        if log_scale:
            y_range_A = np.log10(self.figure_result_data['A'].values.tolist())
            y_range_D = np.log10(self.figure_result_data['D'].values.tolist())
            y_range_E = np.log10(self.figure_result_data['E'].values.tolist())
            y_range_ME = np.log10(self.figure_result_data['ME'].values.tolist())
        else:
            y_range_A = self.figure_result_data['A'].values.tolist()
            y_range_D = self.figure_result_data['D'].values.tolist()
            y_range_E = self.figure_result_data['E'].values.tolist()
            y_range_ME = self.figure_result_data['ME'].values.tolist()
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        ax.plot(x_range, y_range_A)
        ax.scatter(x_range, y_range_A)
        ax.set_ylabel('$log_{10}$ Trace')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - A optimality')
        plt.pyplot.show()
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        ax.plot(x_range, y_range_D)
        ax.scatter(x_range, y_range_D)
        ax.set_ylabel('$log_{10}$ Determinant')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - D optimality')
        plt.pyplot.show()
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        ax.plot(x_range, y_range_E)
        ax.scatter(x_range, y_range_E)
        ax.set_ylabel('$log_{10}$ Minimal eigenvalue')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - E optimality')
        plt.pyplot.show()
        fig = plt.pyplot.figure()
        plt.pyplot.rc('axes', titlesize=font_axes)
        plt.pyplot.rc('axes', labelsize=font_axes)
        plt.pyplot.rc('xtick', labelsize=font_tick)
        plt.pyplot.rc('ytick', labelsize=font_tick)
        ax = fig.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        ax.plot(x_range, y_range_ME)
        ax.scatter(x_range, y_range_ME)
        ax.set_ylabel('$log_{10}$ Condition number')
        ax.set_xlabel(xlabel_text)
        plt.pyplot.title(title_text + ' - Modified E optimality')
        plt.pyplot.show()

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