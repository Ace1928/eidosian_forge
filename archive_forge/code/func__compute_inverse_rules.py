from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def _compute_inverse_rules(self, rules):
    """
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        Arguments:
            rules (dictionary): Rules for which the inverse rules are to computed.

        Returns:
            Dictionary of inverse_rules.

        """
    inverse_rules = {}
    for r in rules:
        rule_key_inverse = r ** (-1)
        rule_value_inverse = rules[r] ** (-1)
        if rule_value_inverse < rule_key_inverse:
            inverse_rules[rule_key_inverse] = rule_value_inverse
        else:
            inverse_rules[rule_value_inverse] = rule_key_inverse
    return inverse_rules