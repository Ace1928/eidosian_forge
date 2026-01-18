import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
class AverageNetworks(BaseInterface):
    """
    Calculates and outputs the average network given a set of input NetworkX gpickle files

    This interface will only keep an edge in the averaged network if that edge is present in
    at least half of the input networks.

    Example
    -------
    >>> import nipype.interfaces.cmtk as cmtk
    >>> avg = cmtk.AverageNetworks()
    >>> avg.inputs.in_files = ['subj1.pck', 'subj2.pck']
    >>> avg.run()                 # doctest: +SKIP

    """
    input_spec = AverageNetworksInputSpec
    output_spec = AverageNetworksOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.resolution_network_file):
            ntwk_res_file = self.inputs.resolution_network_file
        else:
            ntwk_res_file = self.inputs.in_files[0]
        global matlab_network_list
        network_name, matlab_network_list = average_networks(self.inputs.in_files, ntwk_res_file, self.inputs.group_id)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_gpickled_groupavg):
            outputs['gpickled_groupavg'] = op.abspath(self._gen_outfilename(self.inputs.group_id + '_average', 'pck'))
        else:
            outputs['gpickled_groupavg'] = op.abspath(self.inputs.out_gpickled_groupavg)
        if not isdefined(self.inputs.out_gexf_groupavg):
            outputs['gexf_groupavg'] = op.abspath(self._gen_outfilename(self.inputs.group_id + '_average', 'gexf'))
        else:
            outputs['gexf_groupavg'] = op.abspath(self.inputs.out_gexf_groupavg)
        outputs['matlab_groupavgs'] = matlab_network_list
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext