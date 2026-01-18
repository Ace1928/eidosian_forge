from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
def _allowed_transition_names_and_token_types(self):

    def iterate():
        for stack_node in reversed(self):
            for transition in stack_node.dfa.transitions:
                if isinstance(transition, ReservedString):
                    yield transition.value
                else:
                    yield transition
            if not stack_node.dfa.is_final:
                break
    return list(iterate())