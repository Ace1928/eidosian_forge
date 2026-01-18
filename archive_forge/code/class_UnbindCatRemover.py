import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
class UnbindCatRemover(SplitCatSimplifier):
    """
    Helper class to merge Unbind->Cat/Stack. Many of the cases are similar to SplitCatSimplifier.

    Unbind can't be simplified like splits. So, we can only remove the unbind node. Other than this,
    other cases like multiple users, additional args, dim mismatch are similar to `SplitCatSimplifier`,
    hence we extend that class.
    """

    def remove_unbind(self, graph: torch.fx.Graph, unbind_node: torch.fx.Node):
        num_unbind = max((getitem_node.args[1] for getitem_node in unbind_node.users.keys())) + 1
        split_sections = [1 for _ in range(num_unbind)]
        super().simplify(graph, unbind_node, split_sections)

    def get_simplified_split_ranges(self, split_sections: List[int], next_users: List[torch.fx.Node], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[_Range]]:
        simplified_split_ranges = super().get_simplified_split_ranges(split_sections, next_users, user_inputs_list)
        if not simplified_split_ranges or len(simplified_split_ranges) != 1:
            return None
        return simplified_split_ranges

    def get_transform_params(self, unbind_node: torch.fx.Node, next_users: List[torch.fx.Node], user_inputs_list: List[List[Union[torch.fx.Node, _Range]]]) -> Optional[List[List[_TransformParam]]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """
        split_dim = unbind_node.kwargs['dim']
        transform_params_list: List[List[_TransformParam]] = []
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            cat_dim = get_arg_value(user_node, 1, 'dim') or 0
            transform_params: List[_TransformParam] = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    movedim_params = (split_dim, cat_dim) if split_dim != cat_dim else None
                    flatten_params = None
                    if user_node.target == torch.cat:
                        flatten_params = (cat_dim, cat_dim + 1)
                    transform_params.append((None, movedim_params, None, flatten_params))
                elif user_node.target == torch.stack:
                    transform_params.append((None, None, (cat_dim,), None))
                else:
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list