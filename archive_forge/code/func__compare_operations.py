import re
def _compare_operations(self, root_a, root_b):
    """
        Compare execution plan operation tree

        Return: True if operation trees are equal, False otherwise
        """
    if root_a != root_b:
        return False
    if root_a.child_count() != root_b.child_count():
        return False
    for i in range(root_a.child_count()):
        if not self._compare_operations(root_a.children[i], root_b.children[i]):
            return False
    return True