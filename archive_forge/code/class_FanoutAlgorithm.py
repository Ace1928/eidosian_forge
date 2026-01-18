from collections import defaultdict
class FanoutAlgorithm:

    def __init__(self, nodes, edges, verbose=False):
        self.nodes = nodes
        self.edges = edges
        self.rev_edges = make_predecessor_map(edges)
        self.print = print if verbose else self._null_print

    def run(self):
        return self.find_fanout_in_function()

    def _null_print(self, *args, **kwargs):
        pass

    def find_fanout_in_function(self):
        got = {}
        for cur_node in self.edges:
            for incref in (x for x in self.nodes[cur_node] if x == 'incref'):
                decref_blocks = self.find_fanout(cur_node)
                self.print('>>', cur_node, '===', decref_blocks)
                got[cur_node] = decref_blocks
        return got

    def find_fanout(self, head_node):
        decref_blocks = self.find_decref_candidates(head_node)
        self.print('candidates', decref_blocks)
        if not decref_blocks:
            return None
        if not self.verify_non_overlapping(head_node, decref_blocks, entry=ENTRY):
            return None
        return set(decref_blocks)

    def verify_non_overlapping(self, head_node, decref_blocks, entry):
        self.print('verify_non_overlapping'.center(80, '-'))
        todo = list(decref_blocks)
        while todo:
            cur_node = todo.pop()
            visited = set()
            workstack = [cur_node]
            del cur_node
            while workstack:
                cur_node = workstack.pop()
                self.print('cur_node', cur_node, '|', workstack)
                if cur_node in visited:
                    continue
                if cur_node == entry:
                    self.print('!! failed because we arrived at entry', cur_node)
                    return False
                visited.add(cur_node)
                self.print(f'   {cur_node} preds {self.get_predecessors(cur_node)}')
                for pred in self.get_predecessors(cur_node):
                    if pred in decref_blocks:
                        self.print('!! reject because predecessor in decref_blocks')
                        return False
                    if pred != head_node:
                        workstack.append(pred)
        return True

    def get_successors(self, node):
        return tuple(self.edges[node])

    def get_predecessors(self, node):
        return tuple(self.rev_edges[node])

    def has_decref(self, node):
        return 'decref' in self.nodes[node]

    def walk_child_for_decref(self, cur_node, path_stack, decref_blocks, depth=10):
        indent = ' ' * len(path_stack)
        self.print(indent, 'walk', path_stack, cur_node)
        if depth <= 0:
            return False
        if cur_node in path_stack:
            if cur_node == path_stack[0]:
                return False
            return True
        if self.has_decref(cur_node):
            decref_blocks.add(cur_node)
            self.print(indent, 'found decref')
            return True
        depth -= 1
        path_stack += (cur_node,)
        found = False
        for child in self.get_successors(cur_node):
            if not self.walk_child_for_decref(child, path_stack, decref_blocks):
                found = False
                break
            else:
                found = True
        self.print(indent, f'ret {found}')
        return found

    def find_decref_candidates(self, cur_node):
        self.print('find_decref_candidates'.center(80, '-'))
        path_stack = (cur_node,)
        found = False
        decref_blocks = set()
        for child in self.get_successors(cur_node):
            if not self.walk_child_for_decref(child, path_stack, decref_blocks):
                found = False
                break
            else:
                found = True
        if not found:
            return set()
        else:
            return decref_blocks