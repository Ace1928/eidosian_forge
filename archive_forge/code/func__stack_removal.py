from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser
def _stack_removal(self, start_index):
    all_nodes = [node for stack_node in self.stack[start_index:] for node in stack_node.nodes]
    if all_nodes:
        node = tree.PythonErrorNode(all_nodes)
        self.stack[start_index - 1].nodes.append(node)
    self.stack[start_index:] = []
    return bool(all_nodes)