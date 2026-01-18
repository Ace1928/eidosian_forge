import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
class TorchGraph(wandb.data_types.Graph):

    def __init__(self):
        super().__init__('torch')
        self._graph_hooks = set()

    @classmethod
    def hook_torch(cls, model, criterion=None, graph_idx=0):
        wandb.termlog('logging graph, to disable use `wandb.watch(log_graph=False)`')
        graph = TorchGraph()
        graph.hook_torch_modules(model, criterion, graph_idx=graph_idx)
        return graph

    def create_forward_hook(self, name, graph_idx):
        graph = self

        def after_forward_hook(module, input, output):
            if id(module) not in self._graph_hooks:
                return
            if not isinstance(output, tuple):
                output = (output,)
            parameters = [(pname, list(param.size())) for pname, param in module.named_parameters()]
            node = Node(id=id(module), name=name, class_name=str(module), output_shape=nested_shape(output), parameters=parameters, num_parameters=[reduce(mul, size, 1) for pname, size in parameters])
            graph.nodes_by_id[id(module)] = node
            for param in module.parameters():
                graph.nodes_by_id[id(param)] = node
            graph.add_node(node)
            if not graph.criterion_passed:
                if hasattr(output[0], 'grad_fn'):
                    graph.criterion = output[0].grad_fn
                elif isinstance(output[0], list) and output[0] and hasattr(output[0][0], 'grad_fn'):
                    graph.criterion = output[0][0].grad_fn
            self._graph_hooks -= {id(module)}
            if not self._graph_hooks:
                wandb.run.summary['graph_%i' % graph_idx] = self
        return after_forward_hook

    def hook_torch_modules(self, module, criterion=None, prefix=None, graph_idx=0, parent=None):
        torch = util.get_module('torch', 'Could not import torch')
        layers = 0
        graph = self
        if hasattr(module, '_wandb_watch_called') and module._wandb_watch_called:
            raise ValueError('You can only call `wandb.watch` once per model.  Pass a new instance of the model if you need to call wandb.watch again in your code.')
        module._wandb_watch_called = True
        if criterion:
            graph.criterion = criterion
            graph.criterion_passed = True
        for name, sub_module in module.named_children():
            name = name or str(layers)
            if prefix:
                name = prefix + '.' + name
            layers += 1
            if not isinstance(sub_module, torch.nn.Module):
                break
            module_types = [getattr(torch.nn, module_classname) for module_classname in ('Container', 'Sequential', 'ModuleList', 'ModuleDict') if hasattr(torch.nn, module_classname)]
            if parent is None:
                parent = module
            if isinstance(sub_module, tuple(module_types)):
                self.hook_torch_modules(sub_module, prefix=name, parent=parent)
            else:
                self._graph_hooks |= {id(sub_module)}
                try:
                    graph_hook = sub_module.register_forward_hook(self.create_forward_hook(name, graph_idx))
                    wandb.run._torch._hook_handles['topology/' + str(id(graph_hook))] = graph_hook
                    if not hasattr(parent, '_wandb_hook_names'):
                        parent._wandb_hook_names = []
                    parent._wandb_hook_names.append('topology/' + str(id(graph_hook)))
                except RuntimeError as e:
                    wandb.termwarn(f'Trying to register forward_hook failed ({e}) - skipping graph tracking.', repeat=False)

    @classmethod
    def from_torch_layers(cls, module_graph, variable):
        """Recover something like neural net layers from PyTorch Module's and the
        compute graph from a Variable.

        Example output for a multi-layer RNN. We confusingly assign shared embedding values
        to the encoder, but ordered next to the decoder.

        rnns.0.linear.module.weight_raw rnns.0
        rnns.0.linear.module.bias rnns.0
        rnns.1.linear.module.weight_raw rnns.1
        rnns.1.linear.module.bias rnns.1
        rnns.2.linear.module.weight_raw rnns.2
        rnns.2.linear.module.bias rnns.2
        rnns.3.linear.module.weight_raw rnns.3
        rnns.3.linear.module.bias rnns.3
        decoder.weight encoder
        decoder.bias decoder
        """
        torch = util.get_module('torch', 'Could not import torch')
        module_nodes_by_hash = {id(n): n for n in module_graph.nodes}
        module_parameter_nodes = [n for n in module_graph.nodes if isinstance(n.obj, torch.nn.Parameter)]
        names_by_pid = {id(n.obj): n.name for n in module_parameter_nodes}
        reachable_param_nodes = module_graph[0].reachable_descendents()
        reachable_params = {}
        module_reachable_params = {}
        names = {}
        for pid, reachable_nodes in reachable_param_nodes.items():
            node = module_nodes_by_hash[pid]
            if not isinstance(node.obj, torch.nn.Module):
                continue
            module = node.obj
            reachable_params = {}
            module_reachable_params[id(module)] = reachable_params
            names[node.name] = set()
            for reachable_hash in reachable_nodes:
                reachable = module_nodes_by_hash[reachable_hash]
                if isinstance(reachable.obj, torch.nn.Parameter):
                    param = reachable.obj
                    reachable_params[id(param)] = param
                    names[node.name].add(names_by_pid[id(param)])
        node_depths = {id(n): d for n, d in module_graph[0].descendent_bfs()}
        parameter_module_names = {}
        parameter_modules = {}
        for param_node in (n for n in module_graph.nodes if isinstance(n.obj, torch.nn.Parameter)):
            pid = id(param_node.obj)
            best_node = None
            best_depth = None
            best_reachable_params = None
            for node in module_graph.nodes:
                if not isinstance(node.obj, torch.nn.Module):
                    continue
                module = node.obj
                reachable_params = module_reachable_params[id(module)]
                if pid in reachable_params:
                    depth = node_depths[id(node)]
                    if best_node is None or (len(reachable_params), depth) <= (len(best_reachable_params), best_depth):
                        best_node = node
                        best_depth = depth
                        best_reachable_params = reachable_params
            parameter_modules[pid] = best_node
            parameter_module_names[param_node.name] = best_node.name
        reduced_module_graph = cls()
        rmg_ids = itertools.count()
        rmg_root = Node(id=next(rmg_ids), node=module_graph[0])
        reduced_module_graph.add_node(rmg_root)
        reduced_module_graph.root = rmg_root
        rmg_nodes_by_pid = {}
        module_nodes_by_pid = {id(n.obj): n for n in module_graph.nodes}
        compute_graph, compute_node_vars = cls.from_torch_compute_graph(variable)
        for node, _ in reversed(list(compute_graph[0].ancestor_bfs())):
            param = compute_node_vars.get(node.id)
            pid = id(param)
            if not isinstance(param, torch.nn.Parameter):
                continue
            if pid not in module_nodes_by_pid:
                continue
            mid = id(parameter_modules[pid].obj)
            if mid in rmg_nodes_by_pid:
                rmg_module = rmg_nodes_by_pid[mid]
            else:
                rmg_module = rmg_nodes_by_pid[mid] = Node(id=next(rmg_ids), node=module_nodes_by_pid[mid])
                reduced_module_graph.add_node(rmg_module)
                reduced_module_graph.add_edge(rmg_root, rmg_module)
            rmg_param = Node(id=next(rmg_ids), node=module_nodes_by_pid[pid])
            rmg_nodes_by_pid[pid] = rmg_param
            reduced_module_graph.add_node(rmg_param)
            reduced_module_graph.add_edge(rmg_module, rmg_param)
        return reduced_module_graph

    @classmethod
    def node_from_module(cls, nid, module):
        numpy = util.get_module('numpy', 'Could not import numpy')
        node = wandb.Node()
        node.id = nid
        node.child_parameters = 0
        for parameter in module.parameters():
            node.child_parameters += numpy.prod(parameter.size())
        node.class_name = type(module).__name__
        return node