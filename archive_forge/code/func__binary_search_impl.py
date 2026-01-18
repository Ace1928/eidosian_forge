import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _binary_search_impl(self, all_nodes: NodeList, start_idx: int, end_idx: int) -> NodeSet:
    """
        Recursive binary search implementation.
        """
    nodes: NodeList = all_nodes[start_idx:end_idx]
    report: List[str] = []
    self.reports.append(report)
    self.iteration += 1
    report.append(f'Binary search iteration {self.iteration}.')
    report.append(f'From node index {start_idx} to {end_idx - 1}. Size of the interested node list is {len(nodes)}')
    cur_nodes: NodeSet = set(nodes)
    for node in nodes:
        if node in self.fusions:
            cur_nodes.update(self.fusions[node])
    try:
        split_module, submod_name = self._build_submodule(cur_nodes)
        self._run_and_compare(split_module, submod_name, [])
    except (FxNetMinimizerRunFuncError, FxNetMinimizerResultMismatchError):
        if len(nodes) == 1:
            report.append(f'This is the last node in the sub-module. Search in the current branch is successful with culprit = {cur_nodes}.')
            self.print_report(report)
            return cur_nodes
        report.append('Proceed to split and lower the halves of the current sub-module individually.')
        self.print_report(report)
        mid = len(nodes) // 2
        culprits = self._binary_search_impl(all_nodes, start_idx, start_idx + mid)
        if len(culprits) != 0 and (not self.settings.find_all):
            return culprits
        culprits = self._binary_search_impl(all_nodes, start_idx + mid, end_idx)
        if len(culprits) == 0:
            report.append(f'Further split and lowering found no errors. Unable to minimize the submodule with list of nodes: {nodes}')
            self.print_report(report)
        return culprits
    else:
        report.append('No discrepancy found.')
        self.print_report(report)
        return set()