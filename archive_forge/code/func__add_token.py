from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
def _add_token(self, token):
    """
        This is the only core function for parsing. Here happens basically
        everything. Everything is well prepared by the parser generator and we
        only apply the necessary steps here.
        """
    grammar = self._pgen_grammar
    stack = self.stack
    type_, value, start_pos, prefix = token
    transition = _token_to_transition(grammar, type_, value)
    while True:
        try:
            plan = stack[-1].dfa.transitions[transition]
            break
        except KeyError:
            if stack[-1].dfa.is_final:
                self._pop()
            else:
                self.error_recovery(token)
                return
        except IndexError:
            raise InternalParseError('too much input', type_, value, start_pos)
    stack[-1].dfa = plan.next_dfa
    for push in plan.dfa_pushes:
        stack.append(StackNode(push))
    leaf = self.convert_leaf(type_, value, prefix, start_pos)
    stack[-1].nodes.append(leaf)