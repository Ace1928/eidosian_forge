import os
from .base import PluginBase, logger, report_crash, report_nodes_not_run, str2bool
from ..engine.utils import topological_sort
class LinearPlugin(PluginBase):
    """Execute workflow in series"""

    def run(self, graph, config, updatehash=False):
        """Executes a pre-defined pipeline in a serial order.

        Parameters
        ----------

        graph : networkx digraph
            defines order of execution
        """
        import networkx as nx
        try:
            dfs_preorder = nx.dfs_preorder
        except AttributeError:
            dfs_preorder = nx.dfs_preorder_nodes
        if not isinstance(graph, nx.DiGraph):
            raise ValueError('Input must be a networkx digraph object')
        logger.info('Running serially.')
        old_wd = os.getcwd()
        notrun = []
        donotrun = []
        stop_on_first_crash = str2bool(config['execution']['stop_on_first_crash'])
        errors = []
        nodes, _ = topological_sort(graph)
        for node in nodes:
            endstatus = 'end'
            try:
                if node in donotrun:
                    continue
                if self._status_callback:
                    self._status_callback(node, 'start')
                node.run(updatehash=updatehash)
            except Exception as exc:
                endstatus = 'exception'
                crashfile = report_crash(node)
                subnodes = [s for s in dfs_preorder(graph, node)]
                notrun.append({'node': node, 'dependents': subnodes, 'crashfile': crashfile})
                donotrun.extend(subnodes)
                errors.append(exc)
                if stop_on_first_crash:
                    break
            finally:
                if self._status_callback:
                    self._status_callback(node, endstatus)
        os.chdir(old_wd)
        report_nodes_not_run(notrun)
        if errors:
            error, cause = (errors[0], None)
            if isinstance(error, str):
                error = RuntimeError(error)
            if len(errors) > 1:
                error, cause = (RuntimeError(f'{len(errors)} raised. Re-raising first.'), error)
            raise error from cause