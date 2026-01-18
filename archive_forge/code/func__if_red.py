@staticmethod
def _if_red(node):
    if node and node.isRed:
        return node
    return None