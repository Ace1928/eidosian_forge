import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
class GraphPluginBase(PluginBase):
    """Base class for plugins that distribute graphs to workflows"""

    def __init__(self, plugin_args=None):
        if plugin_args and plugin_args.get('status_callback'):
            logger.warning('status_callback not supported for Graph submission plugins')
        super(GraphPluginBase, self).__init__(plugin_args=plugin_args)

    def run(self, graph, config, updatehash=False):
        import networkx as nx
        pyfiles = []
        dependencies = {}
        self._config = config
        nodes = list(nx.topological_sort(graph))
        logger.debug('Creating executable python files for each node')
        for idx, node in enumerate(nodes):
            pyfiles.append(create_pyscript(node, updatehash=updatehash, store_exception=False))
            dependencies[idx] = [nodes.index(prevnode) for prevnode in list(graph.predecessors(node))]
        self._submit_graph(pyfiles, dependencies, nodes)

    def _get_args(self, node, keywords):
        values = ()
        for keyword in keywords:
            value = getattr(self, '_' + keyword)
            if keyword == 'template' and os.path.isfile(value):
                with open(value) as f:
                    value = f.read()
            if hasattr(node, 'plugin_args') and isinstance(node.plugin_args, dict) and (keyword in node.plugin_args):
                if keyword == 'template' and os.path.isfile(node.plugin_args[keyword]):
                    with open(node.plugin_args[keyword]) as f:
                        tmp_value = f.read()
                else:
                    tmp_value = node.plugin_args[keyword]
                if 'overwrite' in node.plugin_args and node.plugin_args['overwrite']:
                    value = tmp_value
                else:
                    value += tmp_value
            values += (value,)
        return values

    def _submit_graph(self, pyfiles, dependencies, nodes):
        """
        pyfiles: list of files corresponding to a topological sort
        dependencies: dictionary of dependencies based on the toplogical sort
        """
        raise NotImplementedError

    def _get_result(self, taskid):
        if taskid not in self._pending:
            raise Exception('Task %d not found' % taskid)
        if self._is_pending(taskid):
            return None
        node_dir = self._pending[taskid]
        glob(os.path.join(node_dir, 'result_*.pklz')).pop()
        results_file = glob(os.path.join(node_dir, 'result_*.pklz'))[0]
        result_data = load_resultfile(results_file)
        result_out = dict(result=None, traceback=None)
        if isinstance(result_data, dict):
            result_out['result'] = result_data['result']
            result_out['traceback'] = result_data['traceback']
            result_out['hostname'] = result_data['hostname']
            if results_file:
                crash_file = os.path.join(node_dir, 'crashstore.pklz')
                os.rename(results_file, crash_file)
        else:
            result_out['result'] = result_data
        return result_out