from types import GenericAlias
def _find_cycle(self):
    n2i = self._node2info
    stack = []
    itstack = []
    seen = set()
    node2stacki = {}
    for node in n2i:
        if node in seen:
            continue
        while True:
            if node in seen:
                if node in node2stacki:
                    return stack[node2stacki[node]:] + [node]
            else:
                seen.add(node)
                itstack.append(iter(n2i[node].successors).__next__)
                node2stacki[node] = len(stack)
                stack.append(node)
            while stack:
                try:
                    node = itstack[-1]()
                    break
                except StopIteration:
                    del node2stacki[stack.pop()]
                    itstack.pop()
            else:
                break
    return None