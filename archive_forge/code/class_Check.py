from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class Check(NodeVisitor):
    """
    Checker for ast <-> pattern.

    NodeVisitor is needed for specific behavior checker.

    Attributes
    ----------
    node : AST
        node we want to compare with pattern
    placeholders : [AST]
        list of placeholder value for later comparison or replacement.
    """

    def __init__(self, node, placeholders):
        """ Initialize attributes. """
        self.node = node
        self.placeholders = placeholders

    def check_list(self, node_list, pattern_list):
        """ Check if list of node are equal. """
        if len(node_list) != len(pattern_list):
            return False
        return all((Check(node_elt, self.placeholders).visit(pattern_elt) for node_elt, pattern_elt in zip(node_list, pattern_list)))

    def visit_Placeholder(self, pattern):
        """
        Save matching node or compare it with the existing one.

        FIXME : What if the new placeholder is a better choice?
        """
        if pattern.id in self.placeholders and (not Check(self.node, self.placeholders).visit(self.placeholders[pattern.id])):
            return False
        elif pattern.type is not None and (not isinstance(self.node, pattern.type)):
            return False
        elif pattern.constraint is not None and (not pattern.constraint(self.node)):
            return False
        else:
            self.placeholders[pattern.id] = self.node
            return True

    @staticmethod
    def visit_AST_any(_):
        """ Every node match with it. """
        return True

    def visit_AST_or(self, pattern):
        """ Match if any of the or content match with the other node. """
        return any((self.field_match(self.node, value_or) for value_or in pattern.args))

    def visit_Set(self, pattern):
        """ Set have unordered values. """
        if not isinstance(self.node, Set):
            return False
        if len(pattern.elts) > MAX_UNORDERED_LENGTH:
            raise DamnTooLongPattern('Pattern for Set is too long')
        return any((self.check_list(self.node.elts, pattern_elts) for pattern_elts in permutations(pattern.elts)))

    def visit_Dict(self, pattern):
        """ Dict can match with unordered values. """
        if not isinstance(self.node, Dict):
            return False
        if len(pattern.keys) > MAX_UNORDERED_LENGTH:
            raise DamnTooLongPattern('Pattern for Dict is too long')
        for permutation in permutations(range(len(self.node.keys))):
            for i, value in enumerate(permutation):
                if not self.field_match(self.node.keys[i], pattern.keys[value]):
                    break
            else:
                pattern_values = [pattern.values[i] for i in permutation]
                return self.check_list(self.node.values, pattern_values)
        return False

    def field_match(self, node_field, pattern_field):
        """
        Check if two fields match.

        Field match if:
            - If it is a list, all values have to match.
            - If if is a node, recursively check it.
            - Otherwise, check values are equal.
        """
        if isinstance(pattern_field, list):
            return self.check_list(node_field, pattern_field)
        if isinstance(pattern_field, AST):
            return Check(node_field, self.placeholders).visit(pattern_field)
        return Check.strict_eq(pattern_field, node_field)

    @staticmethod
    def strict_eq(f0, f1):
        if f0 == f1:
            return True
        try:
            return isnan(f0) and isnan(f1)
        except TypeError:
            return False

    def generic_visit(self, pattern):
        """
        Check if the pattern match with the checked node.

        a node match if:
            - type match
            - all field match
        """
        if not isinstance(pattern, type(self.node)):
            return False
        return all((self.field_match(value, getattr(pattern, field)) for field, value in iter_fields(self.node)))