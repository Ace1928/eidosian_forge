from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
class FastMachine(object):
    """
    FastMachine is a deterministic machine represented in a way that
    allows fast scanning.
    """

    def __init__(self):
        self.initial_states = {}
        self.states = []
        self.next_number = 1
        self.new_state_template = {'': None, 'bol': None, 'eol': None, 'eof': None, 'else': None}

    def __del__(self):
        for state in self.states:
            state.clear()

    def new_state(self, action=None):
        number = self.next_number
        self.next_number = number + 1
        result = self.new_state_template.copy()
        result['number'] = number
        result['action'] = action
        self.states.append(result)
        return result

    def make_initial_state(self, name, state):
        self.initial_states[name] = state

    @cython.locals(code0=cython.int, code1=cython.int, maxint=cython.int, state=dict)
    def add_transitions(self, state, event, new_state, maxint=maxint):
        if type(event) is tuple:
            code0, code1 = event
            if code0 == -maxint:
                state['else'] = new_state
            elif code1 != maxint:
                while code0 < code1:
                    state[unichr(code0)] = new_state
                    code0 += 1
        else:
            state[event] = new_state

    def get_initial_state(self, name):
        return self.initial_states[name]

    def dump(self, file):
        file.write('Plex.FastMachine:\n')
        file.write('   Initial states:\n')
        for name, state in sorted(self.initial_states.items()):
            file.write('      %s: %s\n' % (repr(name), state['number']))
        for state in self.states:
            self.dump_state(state, file)

    def dump_state(self, state, file):
        file.write('   State %d:\n' % state['number'])
        self.dump_transitions(state, file)
        action = state['action']
        if action is not None:
            file.write('      %s\n' % action)

    def dump_transitions(self, state, file):
        chars_leading_to_state = {}
        special_to_state = {}
        for c, s in state.items():
            if len(c) == 1:
                chars = chars_leading_to_state.get(id(s), None)
                if chars is None:
                    chars = []
                    chars_leading_to_state[id(s)] = chars
                chars.append(c)
            elif len(c) <= 4:
                special_to_state[c] = s
        ranges_to_state = {}
        for state in self.states:
            char_list = chars_leading_to_state.get(id(state), None)
            if char_list:
                ranges = self.chars_to_ranges(char_list)
                ranges_to_state[ranges] = state
        for ranges in sorted(ranges_to_state):
            key = self.ranges_to_string(ranges)
            state = ranges_to_state[ranges]
            file.write('      %s --> State %d\n' % (key, state['number']))
        for key in ('bol', 'eol', 'eof', 'else'):
            state = special_to_state.get(key, None)
            if state:
                file.write('      %s --> State %d\n' % (key, state['number']))

    @cython.locals(char_list=list, i=cython.Py_ssize_t, n=cython.Py_ssize_t, c1=cython.long, c2=cython.long)
    def chars_to_ranges(self, char_list):
        char_list.sort()
        i = 0
        n = len(char_list)
        result = []
        while i < n:
            c1 = ord(char_list[i])
            c2 = c1
            i += 1
            while i < n and ord(char_list[i]) == c2 + 1:
                i += 1
                c2 += 1
            result.append((chr(c1), chr(c2)))
        return tuple(result)

    def ranges_to_string(self, range_list):
        return ','.join(map(self.range_to_string, range_list))

    def range_to_string(self, range_tuple):
        c1, c2 = range_tuple
        if c1 == c2:
            return repr(c1)
        else:
            return '%s..%s' % (repr(c1), repr(c2))