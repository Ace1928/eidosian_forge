from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
class _IRNode(abc.ABC):
    """Base class for IR nodes.

    IR nodes are used for Modularize pass only. They add a layer of abstraction on top of
    torch.fx.Node.

    [NOTE: Modularize Pass Implementation]
    The main job of the pass is to group `fx.Node`s that belong to the same `nn.Module`
    forward call, and then create `call_module` node and sub `fx.GraphModule` from them.
    Each `fx.Node` possesses an `nn_module_stack` meta data that contains information
    about the module call stack. See `_ModuleStackMeta` for examples.

    Analysis step
    -------------

    Each module call is identified by a set of base stack layers. For each module call,
    the pass creates a `_ModuleNode` and groups the sequence of nodes that shares the
    same base stack layers.

    For example,

        stack_of_node_0 = [GPT, block0]
        stack_of_node_1 = [GPT, block1]
        stack_of_node_2 = [GPT, block1, Attention1, MLP]
        stack_of_node_3 = [GPT, block1, Attention1]
        stack_of_node_4 = [GPT, block2]

    All nodes belong to the `GPT` module call, since they share the base stack layers [GPT].
    [node_1, node_2, node_3] are grouped for `GPT.block1`, because they share the base
    stack layers [GPT, block1]. And [node_2, node_3] for `GPT.block1.Attention1`, [node_0]
    for `GPT.block0`, and [node_4] for `GPT.block2` respectfully.

    After the analysis step, a hierarchical representation is generated.

    For above example, the representation is:

        _ModuleNode(GPT)
            _ModuleNode(block0)
                _LeafNode(node_0)
            _ModuleNode(block1)
                _LeafNode(node_1)
                _ModuleNode(Attention1)
                    _ModuleNode(MLP)
                        _LeafNode(node_2)
                _LeafNode(node_3)
            _ModuleNode(block2)
                _LeafNode(node_4)

    Construction step
    -----------------

    The second step is to build the actual `call_module` node and the sub `fx.GraphModule`.
    This is done recursively from the leaf `_ModuleNode` to the root.

    For example, the first submodule to be built is `GPT.block1.Attention1.MLP`. Below pair
    is generated from `_ModuleNode(MLP)`.

        fx.GraphModule(GPT.block1.Attention1.MLP)
            graph:
                node_2

        new_mlp_node = `call_module[GPT.block1.Attention1.MLP](...)`

    Next, the `GPT.block1.Attention1` submodule is built. Below is generated from
    `_ModuleNode(Attention1)`.

        fx.GraphModule(GPT.block1.Attention1)
            graph:
                new_mlp_node
                node_3

        new_attention1_node = `call_module[GPT.block1.Attention1](...)`

    Until every submodule is built, the new modularized `fx.GraphModule` is generated.

    Alternatives
    ------------

    The current algorithm adopts a top down approach. A bottom up approach is similar.
    In contrast to these two, an alternative flat order approach is also possible, where
    each node is traversed and copied to the corresponding submodule.

    The advantage of the current approach lies in the encapsulation of the fx.GraphModule
    construction for each individual submodule within a single `build_module` method, which
    can be called separately once the analysis phase is completed, making debugging more
    convenient.

    Regarding construction step, an alternative implementation is to utilize `fx.Interpreter`
    for traversing all the nodes under the flattened root module and copying the nodes
    into their respective submodule under construction. This approach is not adopted because

        1. It uses the flat order approach discussed above. This means one cannot individually
    construct a submodule and examine it while debugging.

        2. The graph execution functionality of `fx.Interpreter` is not necessary for the
    purpose of this pass. Ignoring that, `fx.Interpreter.run` achieves the same effect
    as a for loop over all the nodes.
    """

    @property
    @abc.abstractmethod
    def stack_meta(self) -> _ModuleStackMeta:
        """The module stack meta data associated with this node."""
        ...

    @property
    @abc.abstractmethod
    def stack_trace(self) -> Optional[str]:
        """The stack trace associated with this node."""
        ...