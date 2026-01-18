import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class NetworkXMetrics(BaseInterface):
    """
    Calculates and outputs NetworkX-based measures for an input network

    Example
    -------
    >>> import nipype.interfaces.cmtk as cmtk
    >>> nxmetrics = cmtk.NetworkXMetrics()
    >>> nxmetrics.inputs.in_file = 'subj1.pck'
    >>> nxmetrics.run()                 # doctest: +SKIP

    """
    input_spec = NetworkXMetricsInputSpec
    output_spec = NetworkXMetricsOutputSpec

    def _run_interface(self, runtime):
        import scipy.io as sio
        global gpickled, nodentwks, edgentwks, kntwks, matlab
        gpickled = list()
        nodentwks = list()
        edgentwks = list()
        kntwks = list()
        matlab = list()
        ntwk = _read_pickle(self.inputs.in_file)
        calculate_cliques = self.inputs.compute_clique_related_measures
        weighted = self.inputs.treat_as_weighted_graph
        global_measures = compute_singlevalued_measures(ntwk, weighted, calculate_cliques)
        if isdefined(self.inputs.out_global_metrics_matlab):
            global_out_file = op.abspath(self.inputs.out_global_metrics_matlab)
        else:
            global_out_file = op.abspath(self._gen_outfilename('globalmetrics', 'mat'))
        sio.savemat(global_out_file, global_measures, oned_as='column')
        matlab.append(global_out_file)
        node_measures = compute_node_measures(ntwk, calculate_cliques)
        for key in list(node_measures.keys()):
            newntwk = add_node_data(node_measures[key], ntwk)
            out_file = op.abspath(self._gen_outfilename(key, 'pck'))
            with open(out_file, 'wb') as f:
                pickle.dump(newntwk, f, pickle.HIGHEST_PROTOCOL)
            nodentwks.append(out_file)
        if isdefined(self.inputs.out_node_metrics_matlab):
            node_out_file = op.abspath(self.inputs.out_node_metrics_matlab)
        else:
            node_out_file = op.abspath(self._gen_outfilename('nodemetrics', 'mat'))
        sio.savemat(node_out_file, node_measures, oned_as='column')
        matlab.append(node_out_file)
        gpickled.extend(nodentwks)
        edge_measures = compute_edge_measures(ntwk)
        for key in list(edge_measures.keys()):
            newntwk = add_edge_data(edge_measures[key], ntwk)
            out_file = op.abspath(self._gen_outfilename(key, 'pck'))
            with open(out_file, 'wb') as f:
                pickle.dump(newntwk, f, pickle.HIGHEST_PROTOCOL)
            edgentwks.append(out_file)
        if isdefined(self.inputs.out_edge_metrics_matlab):
            edge_out_file = op.abspath(self.inputs.out_edge_metrics_matlab)
        else:
            edge_out_file = op.abspath(self._gen_outfilename('edgemetrics', 'mat'))
        sio.savemat(edge_out_file, edge_measures, oned_as='column')
        matlab.append(edge_out_file)
        gpickled.extend(edgentwks)
        ntwk_measures = compute_network_measures(ntwk)
        for key in list(ntwk_measures.keys()):
            if key == 'k_core':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_core, 'pck'))
            if key == 'k_shell':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_shell, 'pck'))
            if key == 'k_crust':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_crust, 'pck'))
            with open(out_file, 'wb') as f:
                pickle.dump(ntwk_measures[key], f, pickle.HIGHEST_PROTOCOL)
            kntwks.append(out_file)
        gpickled.extend(kntwks)
        out_pickled_extra_measures = op.abspath(self._gen_outfilename(self.inputs.out_pickled_extra_measures, 'pck'))
        dict_measures = compute_dict_measures(ntwk)
        iflogger.info('Saving extra measure file to %s in Pickle format', op.abspath(out_pickled_extra_measures))
        with open(out_pickled_extra_measures, 'w') as fo:
            pickle.dump(dict_measures, fo)
        iflogger.info('Saving MATLAB measures as %s', matlab)
        global dicts
        dicts = list()
        for idx, key in enumerate(dict_measures.keys()):
            for idxd, keyd in enumerate(dict_measures[key].keys()):
                if idxd == 0:
                    nparraykeys = np.array(keyd)
                    nparrayvalues = np.array(dict_measures[key][keyd])
                else:
                    nparraykeys = np.append(nparraykeys, np.array(keyd))
                    values = np.array(dict_measures[key][keyd])
                    nparrayvalues = np.append(nparrayvalues, values)
            nparray = np.vstack((nparraykeys, nparrayvalues))
            out_file = op.abspath(self._gen_outfilename(key, 'mat'))
            npdict = {}
            npdict[key] = nparray
            sio.savemat(out_file, npdict, oned_as='column')
            dicts.append(out_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['k_core'] = op.abspath(self._gen_outfilename(self.inputs.out_k_core, 'pck'))
        outputs['k_shell'] = op.abspath(self._gen_outfilename(self.inputs.out_k_shell, 'pck'))
        outputs['k_crust'] = op.abspath(self._gen_outfilename(self.inputs.out_k_crust, 'pck'))
        outputs['gpickled_network_files'] = gpickled
        outputs['k_networks'] = kntwks
        outputs['node_measure_networks'] = nodentwks
        outputs['edge_measure_networks'] = edgentwks
        outputs['matlab_dict_measures'] = dicts
        outputs['global_measures_matlab'] = op.abspath(self._gen_outfilename('globalmetrics', 'mat'))
        outputs['node_measures_matlab'] = op.abspath(self._gen_outfilename('nodemetrics', 'mat'))
        outputs['edge_measures_matlab'] = op.abspath(self._gen_outfilename('edgemetrics', 'mat'))
        outputs['matlab_matrix_files'] = [outputs['global_measures_matlab'], outputs['node_measures_matlab'], outputs['edge_measures_matlab']]
        outputs['pickled_extra_measures'] = op.abspath(self._gen_outfilename(self.inputs.out_pickled_extra_measures, 'pck'))
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext