from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
class FSM:
    """
        A Finite State Machine or FSM has an alphabet and a set of states. At any
        given moment, the FSM is in one state. When passed a symbol from the
        alphabet, the FSM jumps to another state (or possibly the same state).
        A map (Python dictionary) indicates where to jump.
        One state is nominated as a starting state. Zero or more states are
        nominated as final states. If, after consuming a string of symbols,
        the FSM is in a final state, then it is said to "accept" the string.
        This class also has some pretty powerful methods which allow FSMs to
        be concatenated, alternated between, multiplied, looped (Kleene star
        closure), intersected, and simplified.
        The majority of these methods are available using operator overloads.
    """
    alphabet: Alphabet
    initial: State
    states: Set[State]
    finals: Set[State]
    map: Dict[State, Dict[TransitionKey, State]]

    def __setattr__(self, name, value):
        """Immutability prevents some potential problems."""
        raise Exception('This object is immutable.')

    def __init__(self, alphabet: Alphabet, states, initial, finals, map, *, __no_validation__=False):
        """
            `alphabet` is an iterable of symbols the FSM can be fed.
            `states` is the set of states for the FSM
            `initial` is the initial state
            `finals` is the set of accepting states
            `map` may be sparse (i.e. it may omit transitions). In the case of omitted
            transitions, a non-final "oblivion" state is simulated.
        """
        if not __no_validation__:
            if not isinstance(alphabet, Alphabet):
                raise TypeError('Expected an Alphabet instance')
            if not initial in states:
                raise Exception('Initial state ' + repr(initial) + ' must be one of ' + repr(states))
            if not finals.issubset(states):
                raise Exception('Final states ' + repr(finals) + ' must be a subset of ' + repr(states))
            for state in map.keys():
                for symbol in map[state]:
                    if not map[state][symbol] in states:
                        raise Exception('Transition for state ' + repr(state) + ' and symbol ' + repr(symbol) + ' leads to ' + repr(map[state][symbol]) + ', which is not a state')
        self.__dict__['alphabet'] = alphabet
        self.__dict__['states'] = frozenset(states)
        self.__dict__['initial'] = initial
        self.__dict__['finals'] = frozenset(finals)
        self.__dict__['map'] = map

    def accepts(self, input: str):
        """
            Test whether the present FSM accepts the supplied string (iterable of
            symbols). Equivalently, consider `self` as a possibly-infinite set of
            strings and test whether `string` is a member of it.
            This is actually mainly used for unit testing purposes.
            If `fsm.anything_else` is in your alphabet, then any symbol not in your
            alphabet will be converted to `fsm.anything_else`.
        """
        state = self.initial
        for symbol in input:
            if anything_else in self.alphabet and (not symbol in self.alphabet):
                symbol = anything_else
            transition = self.alphabet[symbol]
            if not (state in self.map and transition in self.map[state]):
                return False
            state = self.map[state][transition]
        return state in self.finals

    def __contains__(self, string):
        """
            This lets you use the syntax `"a" in fsm1` to see whether the string "a"
            is in the set of strings accepted by `fsm1`.
        """
        return self.accepts(string)

    def reduce(self):
        """
            A result by Brzozowski (1963) shows that a minimal finite state machine
            equivalent to the original can be obtained by reversing the original
            twice.
        """
        return self.reversed().reversed()

    def __repr__(self):
        string = 'fsm('
        string += 'alphabet = ' + repr(self.alphabet)
        string += ', states = ' + repr(self.states)
        string += ', initial = ' + repr(self.initial)
        string += ', finals = ' + repr(self.finals)
        string += ', map = ' + repr(self.map)
        string += ')'
        return string

    def __str__(self):
        rows = []
        row = ['', 'name', 'final?']
        row.extend((soft_repr(symbol) for symbol in sorted(self.alphabet)))
        rows.append(row)
        for state in self.states:
            row = []
            if state == self.initial:
                row.append('*')
            else:
                row.append('')
            row.append(str(state))
            if state in self.finals:
                row.append('True')
            else:
                row.append('False')
            for symbol, transition in sorted(self.alphabet.items()):
                if state in self.map and transition in self.map[state]:
                    row.append(str(self.map[state][transition]))
                else:
                    row.append('')
            rows.append(row)
        colwidths = []
        for x in range(len(rows[0])):
            colwidths.append(max((len(str(rows[y][x])) for y in range(len(rows)))) + 1)
        for y in range(len(rows)):
            for x in range(len(rows[y])):
                rows[y][x] = rows[y][x].ljust(colwidths[x])
        rows.insert(1, ['-' * colwidth for colwidth in colwidths])
        return ''.join((''.join(row) + '\n' for row in rows))

    def concatenate(*fsms):
        """
            Concatenate arbitrarily many finite state machines together.
        """
        if len(fsms) == 0:
            return epsilon(Alphabet({}))
        alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])
        last_index, last = (len(fsms) - 1, fsms[-1])

        def connect_all(i, substate):
            """
                Take a state in the numbered FSM and return a set containing it, plus
                (if it's final) the first state from the next FSM, plus (if that's
                final) the first state from the next but one FSM, plus...
            """
            result = {(i, substate)}
            while i < last_index and substate in fsms[i].finals:
                i += 1
                substate = fsms[i].initial
                result.add((i, substate))
            return result
        initial = set()
        if len(fsms) > 0:
            initial.update(connect_all(0, fsms[0].initial))
        initial = frozenset(initial)

        def final(state):
            """If you're in a final state of the final FSM, it's final"""
            for i, substate in state:
                if i == last_index and substate in last.finals:
                    return True
            return False

        def follow(current, new_transition):
            """
                Follow the collection of states through all FSMs at once, jumping to the
                next FSM if we reach the end of the current one
                TODO: improve all follow() implementations to allow for dead metastates?
            """
            next = set()
            for i, substate in current:
                fsm = fsms[i]
                if substate in fsm.map and new_to_old[i][new_transition] in fsm.map[substate]:
                    next.update(connect_all(i, fsm.map[substate][new_to_old[i][new_transition]]))
            if not next:
                raise OblivionError
            return frozenset(next)
        return crawl(alphabet, initial, final, follow)

    def __add__(self, other):
        """
            Concatenate two finite state machines together.
            For example, if self accepts "0*" and other accepts "1+(0|1)",
            will return a finite state machine accepting "0*1+(0|1)".
            Accomplished by effectively following non-deterministically.
            Call using "fsm3 = fsm1 + fsm2"
        """
        return self.concatenate(other)

    def star(self):
        """
            If the present FSM accepts X, returns an FSM accepting X* (i.e. 0 or
            more Xes). This is NOT as simple as naively connecting the final states
            back to the initial state: see (b*ab)* for example.
        """
        alphabet = self.alphabet
        initial = {self.initial}

        def follow(state, transition):
            next = set()
            for substate in state:
                if substate in self.map and transition in self.map[substate]:
                    next.add(self.map[substate][transition])
                if substate in self.finals and self.initial in self.map and (transition in self.map[self.initial]):
                    next.add(self.map[self.initial][transition])
            if not next:
                raise OblivionError
            return frozenset(next)

        def final(state):
            return any((substate in self.finals for substate in state))
        base = crawl(alphabet, initial, final, follow)
        base.__dict__['finals'] = base.finals | {base.initial}
        return base

    def times(self, multiplier):
        """
            Given an FSM and a multiplier, return the multiplied FSM.
        """
        if multiplier < 0:
            raise Exception("Can't multiply an FSM by " + repr(multiplier))
        alphabet = self.alphabet
        initial = {(self.initial, 0)}

        def final(state):
            """If the initial state is final then multiplying doesn't alter that"""
            for substate, iteration in state:
                if substate == self.initial and (self.initial in self.finals or iteration == multiplier):
                    return True
            return False

        def follow(current, transition):
            next = []
            for substate, iteration in current:
                if iteration < multiplier and substate in self.map and (transition in self.map[substate]):
                    next.append((self.map[substate][transition], iteration))
                    if self.map[substate][transition] in self.finals:
                        next.append((self.initial, iteration + 1))
            if len(next) == 0:
                raise OblivionError
            return frozenset(next)
        return crawl(alphabet, initial, final, follow)

    def __mul__(self, multiplier):
        """
            Given an FSM and a multiplier, return the multiplied FSM.
        """
        return self.times(multiplier)

    def union(*fsms):
        """
            Treat `fsms` as a collection of arbitrary FSMs and return the union FSM.
            Can be used as `fsm1.union(fsm2, ...)` or `fsm.union(fsm1, ...)`. `fsms`
            may be empty.
        """
        return parallel(fsms, any)

    def __or__(self, other):
        """
            Alternation.
            Return a finite state machine which accepts any sequence of symbols
            that is accepted by either self or other. Note that the set of strings
            recognised by the two FSMs undergoes a set union.
            Call using "fsm3 = fsm1 | fsm2"
        """
        return self.union(other)

    def intersection(*fsms):
        """
            Intersection.
            Take FSMs and AND them together. That is, return an FSM which
            accepts any sequence of symbols that is accepted by both of the original
            FSMs. Note that the set of strings recognised by the two FSMs undergoes
            a set intersection operation.
            Call using "fsm3 = fsm1 & fsm2"
        """
        return parallel(fsms, all)

    def __and__(self, other):
        """
            Treat the FSMs as sets of strings and return the intersection of those
            sets in the form of a new FSM. `fsm1.intersection(fsm2, ...)` or
            `fsm.intersection(fsm1, ...)` are acceptable.
        """
        return self.intersection(other)

    def symmetric_difference(*fsms):
        """
            Treat `fsms` as a collection of sets of strings and compute the symmetric
            difference of them all. The python set method only allows two sets to be
            operated on at once, but we go the extra mile since it's not too hard.
        """
        return parallel(fsms, lambda accepts: accepts.count(True) % 2 == 1)

    def __xor__(self, other):
        """
            Symmetric difference. Returns an FSM which recognises only the strings
            recognised by `self` or `other` but not both.
        """
        return self.symmetric_difference(other)

    def everythingbut(self):
        """
            Return a finite state machine which will accept any string NOT
            accepted by self, and will not accept any string accepted by self.
            This is more complicated if there are missing transitions, because the
            missing "dead" state must now be reified.
        """
        alphabet = self.alphabet
        initial = {0: self.initial}

        def follow(current, transition):
            next = {}
            if 0 in current and current[0] in self.map and (transition in self.map[current[0]]):
                next[0] = self.map[current[0]][transition]
            return next

        def final(state):
            return not (0 in state and state[0] in self.finals)
        return crawl(alphabet, initial, final, follow)

    def isdisjoint(self, other: 'FSM') -> bool:
        alphabet, new_to_old = self.alphabet.intersect(other.alphabet)
        initial = (self.initial, other.initial)

        def follow(current, transition):
            ss, os = current
            if ss in self.map and new_to_old[0][transition] in self.map[ss]:
                sn = self.map[ss][new_to_old[0][transition]]
            else:
                sn = None
            if os in other.map and new_to_old[1][transition] in other.map[os]:
                on = other.map[os][new_to_old[1][transition]]
            else:
                on = None
            if not sn or not on:
                raise OblivionError
            return (sn, on)

        def final(state):
            if state[0] in self.finals and state[1] in other.finals:
                raise _Marker
        try:
            crawl_hash_no_result(alphabet, initial, final, follow)
        except _Marker:
            return False
        else:
            return True

    def reversed(self):
        """
            Return a new FSM such that for every string that self accepts (e.g.
            "beer", the new FSM accepts the reversed string ("reeb").
        """
        alphabet = self.alphabet
        initial = frozenset(self.finals)
        reverse_map = {}
        for state, transition_map in self.map.items():
            for transition, next_state in transition_map.items():
                if (next_state, transition) not in reverse_map:
                    reverse_map[next_state, transition] = set()
                reverse_map[next_state, transition].add(state)

        def follow(current, transition):
            next_states = set()
            for state in current:
                next_states.update(reverse_map.get((state, transition), set()))
            if not next_states:
                raise OblivionError
            return frozenset(next_states)

        def final(state):
            return self.initial in state
        return crawl(alphabet, initial, final, follow)

    def __reversed__(self):
        """
            Return a new FSM such that for every string that self accepts (e.g.
            "beer", the new FSM accepts the reversed string ("reeb").
        """
        return self.reversed()

    def islive(self, state):
        """A state is "live" if a final state can be reached from it."""
        seen = {state}
        reachable = [state]
        i = 0
        while i < len(reachable):
            current = reachable[i]
            if current in self.finals:
                return True
            if current in self.map:
                for transition in self.map[current]:
                    next = self.map[current][transition]
                    if next not in seen:
                        reachable.append(next)
                        seen.add(next)
            i += 1
        return False

    def empty(self):
        """
            An FSM is empty if it recognises no strings. An FSM may be arbitrarily
            complicated and have arbitrarily many final states while still recognising
            no strings because those final states may all be inaccessible from the
            initial state. Equally, an FSM may be non-empty despite having an empty
            alphabet if the initial state is final.
        """
        return not self.islive(self.initial)

    def strings(self, max_iterations=None):
        """
            Generate strings (lists of symbols) that this FSM accepts. Since there may
            be infinitely many of these we use a generator instead of constructing a
            static list. Strings will be sorted in order of length and then lexically.
            This procedure uses arbitrary amounts of memory but is very fast. There
            may be more efficient ways to do this, that I haven't investigated yet.
            You can use this in list comprehensions.

            `max_iterations` controls how many attempts will be made to generate strings.
            For complex FSM it can take minutes to actually find something.
            If this isn't acceptable, provide a value to `max_iterations`.
            The approximate time complexity is
            0.15 seconds per 10_000 iterations per 10 symbols
        """
        livestates = set((state for state in self.states if self.islive(state)))
        strings = deque()
        cstate = self.initial
        cstring = []
        if cstate in livestates:
            if cstate in self.finals:
                yield cstring
            strings.append((cstring, cstate))
        i = 0
        while strings:
            cstring, cstate = strings.popleft()
            i += 1
            if cstate in self.map:
                for transition in sorted(self.map[cstate]):
                    nstate = self.map[cstate][transition]
                    if nstate in livestates:
                        for symbol in sorted(self.alphabet.by_transition[transition]):
                            nstring = cstring + [symbol]
                            if nstate in self.finals:
                                yield nstring
                            strings.append((nstring, nstate))
            if max_iterations is not None and i > max_iterations:
                raise ValueError(f"Couldn't find an example within {max_iterations} iterations")

    def __iter__(self):
        """
            This allows you to do `for string in fsm1` as a list comprehension!
        """
        return self.strings()

    def equivalent(self, other):
        """
            Two FSMs are considered equivalent if they recognise the same strings.
            Or, to put it another way, if their symmetric difference recognises no
            strings.
        """
        return (self ^ other).empty()

    def __eq__(self, other):
        """
            You can use `fsm1 == fsm2` to determine whether two FSMs recognise the
            same strings.
        """
        return self.equivalent(other)

    def different(self, other):
        """
            Two FSMs are considered different if they have a non-empty symmetric
            difference.
        """
        return not (self ^ other).empty()

    def __ne__(self, other):
        """
            Use `fsm1 != fsm2` to determine whether two FSMs recognise different
            strings.
        """
        return self.different(other)

    def difference(*fsms):
        """
            Difference. Returns an FSM which recognises only the strings
            recognised by the first FSM in the list, but none of the others.
        """
        return parallel(fsms, lambda accepts: accepts[0] and (not any(accepts[1:])))

    def __sub__(self, other):
        return self.difference(other)

    def cardinality(self):
        """
            Consider the FSM as a set of strings and return the cardinality of that
            set, or raise an OverflowError if there are infinitely many
        """
        num_strings = {}

        def get_num_strings(state):
            if self.islive(state):
                if state in num_strings:
                    if num_strings[state] is None:
                        raise OverflowError(state)
                    return num_strings[state]
                num_strings[state] = None
                n = 0
                if state in self.finals:
                    n += 1
                if state in self.map:
                    for transition in self.map[state]:
                        n += get_num_strings(self.map[state][transition]) * len(self.alphabet.by_transition[transition])
                num_strings[state] = n
            else:
                num_strings[state] = 0
            return num_strings[state]
        return get_num_strings(self.initial)

    def __len__(self):
        """
            Consider the FSM as a set of strings and return the cardinality of that
            set, or raise an OverflowError if there are infinitely many
        """
        return self.cardinality()

    def issubset(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a subset
            of `other`... `self` recognises no strings which `other` doesn't.
        """
        return (self - other).empty()

    def __le__(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a subset
            of `other`... `self` recognises no strings which `other` doesn't.
        """
        return self.issubset(other)

    def ispropersubset(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a proper
            subset of `other`.
        """
        return self <= other and self != other

    def __lt__(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a strict
            subset of `other`.
        """
        return self.ispropersubset(other)

    def issuperset(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a
            superset of `other`.
        """
        return (other - self).empty()

    def __ge__(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a
            superset of `other`.
        """
        return self.issuperset(other)

    def ispropersuperset(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a proper
            superset of `other`.
        """
        return self >= other and self != other

    def __gt__(self, other):
        """
            Treat `self` and `other` as sets of strings and see if `self` is a
            strict superset of `other`.
        """
        return self.ispropersuperset(other)

    def copy(self):
        """
            For completeness only, since `set.copy()` also exists. FSM objects are
            immutable, so I can see only very odd reasons to need this.
        """
        return FSM(alphabet=self.alphabet.copy(), states=self.states.copy(), initial=self.initial, finals=self.finals.copy(), map=self.map.copy(), __no_validation__=True)

    def derive(self, input):
        """
            Compute the Brzozowski derivative of this FSM with respect to the input
            string of symbols. <https://en.wikipedia.org/wiki/Brzozowski_derivative>
            If any of the symbols are not members of the alphabet, that's a KeyError.
            If you fall into oblivion, then the derivative is an FSM accepting no
            strings.
        """
        try:
            state = self.initial
            for symbol in input:
                if not symbol in self.alphabet:
                    if not anything_else in self.alphabet:
                        raise KeyError(symbol)
                    symbol = anything_else
                if not (state in self.map and self.alphabet[symbol] in self.map[state]):
                    raise OblivionError
                state = self.map[state][self.alphabet[symbol]]
            return FSM(alphabet=self.alphabet, states=self.states, initial=state, finals=self.finals, map=self.map, __no_validation__=True)
        except OblivionError:
            return null(self.alphabet)