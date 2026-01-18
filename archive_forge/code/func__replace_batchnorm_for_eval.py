import torch
import torch.nn.functional as F
def _replace_batchnorm_for_eval(m: torch.fx.GraphModule):
    from .utils import get_aten_graph_module
    m.graph.eliminate_dead_code()
    m.recompile()

    def bn_train(x: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor):
        return F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)

    def bn_eval(x: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor):
        return F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False)
    example_inputs = (torch.randn(1, 1, 3, 3), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1))
    match_pattern = get_aten_graph_module(bn_train, example_inputs)
    replacement_pattern = get_aten_graph_module(bn_eval, example_inputs)
    from torch.fx.subgraph_rewriter import replace_pattern_with_filters
    replace_pattern_with_filters(m, match_pattern, replacement_pattern, match_filters=[], ignore_literals=True)
    m.recompile()