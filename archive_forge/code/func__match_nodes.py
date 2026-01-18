from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool:
    logger.info('  matching %s to %s', pn, gn)
    assert isinstance(pn, Node) and isinstance(gn, Node), str(f'pn and gn must be Node, pn: {pn}, gn: {gn}')
    if pn in match.nodes_map:
        return match.nodes_map[pn] == gn
    if gn in match.nodes_map.values():
        return False
    if not self._nodes_are_equal(pn, gn):
        return False
    saved_match = copy.copy(match)
    match.nodes_map[pn] = gn
    if pn.op == 'placeholder':
        return True
    match_found = True

    def _match_args(args1: Union[List, Tuple], args2: Union[List, Tuple]) -> bool:
        if len(args1) != len(args2):
            return False
        for a1, a2 in zip(args1, args2):
            if isinstance(a1, Node) and isinstance(a2, Node):
                matched = self._match_nodes(a1, a2, match)
            elif isinstance(a1, (list, tuple)) and isinstance(a2, (list, tuple)):
                matched = _match_args(a1, a2)
            else:
                matched = self._match_literals(a1, a2, match) or self.ignore_literals
            if not matched:
                return False
        return True
    pn_args, gn_args = (None, None)
    if (len(pn.args) != len(gn.args) or list(pn.kwargs.keys()) != list(gn.kwargs.keys())) and pn.op == 'call_function' and isinstance(pn.target, torch._ops.OpOverload):
        args_schema = pn.target._schema.arguments

        def get_all_arguments(orig_args, orig_kwargs):
            all_args = []
            for i, schema in enumerate(args_schema):
                if schema.name in orig_kwargs:
                    all_args.append(orig_kwargs[schema.name])
                elif not schema.kwarg_only and i < len(orig_args):
                    all_args.append(orig_args[i])
                else:
                    all_args.append(schema.default_value)
            return all_args
        pn_args = get_all_arguments(pn.args, pn.kwargs)
        gn_args = get_all_arguments(gn.args, gn.kwargs)
    elif len(pn.args) == len(gn.args) and list(pn.kwargs.keys()) == list(gn.kwargs.keys()):
        pn_args = list(pn.args)
        gn_args = list(gn.args)
        pn_args.extend(list(pn.kwargs.values()))
        gn_args.extend(list(gn.kwargs.values()))
    else:
        match_found = False
    match_found = match_found and pn_args is not None and (gn_args is not None) and _match_args(pn_args, gn_args)
    if not match_found:
        match = copy.copy(saved_match)
        return False
    return True