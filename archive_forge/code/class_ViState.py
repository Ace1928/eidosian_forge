from __future__ import unicode_literals
class ViState(object):
    """
    Mutable class to hold the state of the Vi navigation.
    """

    def __init__(self):
        self.last_character_find = None
        self.operator_func = None
        self.operator_arg = None
        self.named_registers = {}
        self.input_mode = InputMode.INSERT
        self.waiting_for_digraph = False
        self.digraph_symbol1 = None
        self.tilde_operator = False

    def reset(self, mode=InputMode.INSERT):
        """
        Reset state, go back to the given mode. INSERT by default.
        """
        self.input_mode = mode
        self.waiting_for_digraph = False
        self.operator_func = None
        self.operator_arg = None