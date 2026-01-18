import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
class FiniteMachine(object):
    """A finite state machine.

    This state machine can be used to automatically run a given set of
    transitions and states in response to events (either from callbacks or from
    generator/iterator send() values, see PEP 342). On each triggered event, a
    ``on_enter`` and ``on_exit`` callback can also be provided which will be
    called to perform some type of action on leaving a prior state and before
    entering a new state.

    NOTE(harlowja): reactions will *only* be called when the generator/iterator
    from :py:meth:`~automaton.runners.Runner.run_iter` does *not* send
    back a new event (they will always be called if the
    :py:meth:`~automaton.runners.Runner.run` method is used). This allows
    for two unique ways (these ways can also be intermixed) to use this state
    machine when using :py:meth:`~automaton.runners.Runner.run`; one
    where *external* event trigger the next state transition and one
    where *internal* reaction callbacks trigger the next state
    transition. The other way to use this
    state machine is to skip using  :py:meth:`~automaton.runners.Runner.run`
    or :py:meth:`~automaton.runners.Runner.run_iter`
    completely and use the :meth:`~.FiniteMachine.process_event` method
    explicitly and trigger the events via
    some *external* functionality/triggers...
    """
    Effect = collections.namedtuple('Effect', 'reaction,terminal')

    @classmethod
    def _effect_builder(cls, new_state, event):
        return cls.Effect(new_state['reactions'].get(event), new_state['terminal'])

    def __init__(self):
        self._transitions = {}
        self._states = collections.OrderedDict()
        self._default_start_state = None
        self._current = None
        self.frozen = False

    @property
    def default_start_state(self):
        """Sets the *default* start state that the machine should use.

        NOTE(harlowja): this will be used by ``initialize`` but only if that
        function is not given its own ``start_state`` that overrides this
        default.
        """
        return self._default_start_state

    @default_start_state.setter
    def default_start_state(self, state):
        if self.frozen:
            raise excp.FrozenMachine()
        if state not in self._states:
            raise excp.NotFound("Can not set the default start state to undefined state '%s'" % state)
        self._default_start_state = state

    @classmethod
    def build(cls, state_space):
        """Builds a machine from a state space listing.

        Each element of this list must be an instance
        of :py:class:`.State` or a ``dict`` with equivalent keys that
        can be used to construct a :py:class:`.State` instance.
        """
        state_space = list(_convert_to_states(state_space))
        m = cls()
        for state in state_space:
            m.add_state(state.name, terminal=state.is_terminal, on_enter=state.on_enter, on_exit=state.on_exit)
        for state in state_space:
            if state.next_states:
                for event, next_state in state.next_states.items():
                    if isinstance(next_state, State):
                        next_state = next_state.name
                    m.add_transition(state.name, next_state, event)
        return m

    @property
    def current_state(self):
        """The current state the machine is in (or none if not initialized)."""
        if self._current is not None:
            return self._current.name
        return None

    @property
    def terminated(self):
        """Returns whether the state machine is in a terminal state."""
        if self._current is None:
            return False
        return self._states[self._current.name]['terminal']

    def add_state(self, state, terminal=False, on_enter=None, on_exit=None):
        """Adds a given state to the state machine.

        The ``on_enter`` and ``on_exit`` callbacks, if provided will be
        expected to take two positional parameters, these being the state
        being exited (for ``on_exit``) or the state being entered (for
        ``on_enter``) and a second parameter which is the event that is
        being processed that caused the state transition.
        """
        if self.frozen:
            raise excp.FrozenMachine()
        if state in self._states:
            raise excp.Duplicate("State '%s' already defined" % state)
        if on_enter is not None:
            if not callable(on_enter):
                raise ValueError('On enter callback must be callable')
        if on_exit is not None:
            if not callable(on_exit):
                raise ValueError('On exit callback must be callable')
        self._states[state] = {'terminal': bool(terminal), 'reactions': {}, 'on_enter': on_enter, 'on_exit': on_exit}
        self._transitions[state] = collections.OrderedDict()

    def is_actionable_event(self, event):
        """Check whether the event is actionable in the current state."""
        current = self._current
        if current is None:
            return False
        if event not in self._transitions[current.name]:
            return False
        return True

    def add_reaction(self, state, event, reaction, *args, **kwargs):
        """Adds a reaction that may get triggered by the given event & state.

        Reaction callbacks may (depending on how the state machine is ran) be
        used after an event is processed (and a transition occurs) to cause the
        machine to react to the newly arrived at stable state.

        These callbacks are expected to accept three default positional
        parameters (although more can be passed in via *args and **kwargs,
        these will automatically get provided to the callback when it is
        activated *ontop* of the three default). The three default parameters
        are the last stable state, the new stable state and the event that
        caused the transition to this new stable state to be arrived at.

        The expected result of a callback is expected to be a new event that
        the callback wants the state machine to react to. This new event
        may (depending on how the state machine is ran) get processed (and
        this process typically repeats) until the state machine reaches a
        terminal state.
        """
        if self.frozen:
            raise excp.FrozenMachine()
        if state not in self._states:
            raise excp.NotFound("Can not add a reaction to event '%s' for an undefined state '%s'" % (event, state))
        if not callable(reaction):
            raise ValueError('Reaction callback must be callable')
        if event not in self._states[state]['reactions']:
            self._states[state]['reactions'][event] = (reaction, args, kwargs)
        else:
            raise excp.Duplicate("State '%s' reaction to event '%s' already defined" % (state, event))

    def add_transition(self, start, end, event, replace=False):
        """Adds an allowed transition from start -> end for the given event.

        :param start: starting state
        :param end: ending state
        :param event: event that causes start state to
                      transition to end state
        :param replace: replace existing event instead of raising a
                        :py:class:`~automaton.exceptions.Duplicate` exception
                        when the transition already exists.
        """
        if self.frozen:
            raise excp.FrozenMachine()
        if start not in self._states:
            raise excp.NotFound("Can not add a transition on event '%s' that starts in a undefined state '%s'" % (event, start))
        if end not in self._states:
            raise excp.NotFound("Can not add a transition on event '%s' that ends in a undefined state '%s'" % (event, end))
        if self._states[start]['terminal']:
            raise excp.InvalidState("Can not add a transition on event '%s' that starts in the terminal state '%s'" % (event, start))
        if event in self._transitions[start] and (not replace):
            target = self._transitions[start][event]
            if target.name != end:
                raise excp.Duplicate("Cannot add transition from '%(start_state)s' to '%(end_state)s' on event '%(event)s' because a transition from '%(start_state)s' to '%(existing_end_state)s' on event '%(event)s' already exists." % {'existing_end_state': target.name, 'end_state': end, 'event': event, 'start_state': start})
        else:
            target = _Jump(end, self._states[end]['on_enter'], self._states[start]['on_exit'])
            self._transitions[start][event] = target

    def _pre_process_event(self, event):
        current = self._current
        if current is None:
            raise excp.NotInitialized("Can not process event '%s'; the state machine hasn't been initialized" % event)
        if self._states[current.name]['terminal']:
            raise excp.InvalidState("Can not transition from terminal state '%s' on event '%s'" % (current.name, event))
        if event not in self._transitions[current.name]:
            raise excp.NotFound("Can not transition from state '%s' on event '%s' (no defined transition)" % (current.name, event))

    def _post_process_event(self, event, result):
        return result

    def process_event(self, event):
        """Trigger a state change in response to the provided event.

        :returns: Effect this is either a :py:class:`.FiniteMachine.Effect` or
                  an ``Effect`` from a subclass of :py:class:`.FiniteMachine`.
                  See the appropriate named tuple for a description of the
                  actual items in the tuple. For
                  example, :py:class:`.FiniteMachine.Effect`'s
                  first item is ``reaction``: one could invoke this reaction's
                  callback to react to the new stable state.
        :rtype: namedtuple
        """
        self._pre_process_event(event)
        current = self._current
        replacement = self._transitions[current.name][event]
        if current.on_exit is not None:
            current.on_exit(current.name, event)
        if replacement.on_enter is not None:
            replacement.on_enter(replacement.name, event)
        self._current = replacement
        result = self._effect_builder(self._states[replacement.name], event)
        return self._post_process_event(event, result)

    def initialize(self, start_state=None):
        """Sets up the state machine (sets current state to start state...).

        :param start_state: explicit start state to use to initialize the
                            state machine to. If ``None`` is provided then
                            the machine's default start state will be used
                            instead.
        """
        if start_state is None:
            start_state = self._default_start_state
        if start_state not in self._states:
            raise excp.NotFound("Can not start from a undefined state '%s'" % start_state)
        if self._states[start_state]['terminal']:
            raise excp.InvalidState("Can not start from a terminal state '%s'" % start_state)
        self._current = _Jump(start_state, None, self._states[start_state]['on_exit'])

    def copy(self, shallow=False, unfreeze=False):
        """Copies the current state machine.

        NOTE(harlowja): the copy will be left in an *uninitialized* state.

        NOTE(harlowja): when a shallow copy is requested the copy will share
                        the same transition table and state table as the
                        source; this can be advantageous if you have a machine
                        and transitions + states that is defined somewhere
                        and want to use copies to run with (the copies have
                        the current state that is different between machines).
        """
        c = type(self)()
        c._default_start_state = self._default_start_state
        if unfreeze and self.frozen:
            c.frozen = False
        else:
            c.frozen = self.frozen
        if not shallow:
            for state, data in self._states.items():
                copied_data = data.copy()
                copied_data['reactions'] = copied_data['reactions'].copy()
                c._states[state] = copied_data
            for state, data in self._transitions.items():
                c._transitions[state] = data.copy()
        else:
            c._transitions = self._transitions
            c._states = self._states
        return c

    def __contains__(self, state):
        """Returns if this state exists in the machines known states."""
        return state in self._states

    def freeze(self):
        """Freezes & stops addition of states, transitions, reactions..."""
        self.frozen = True

    @property
    def states(self):
        """Returns the state names."""
        return list(self._states)

    @property
    def events(self):
        """Returns how many events exist."""
        c = 0
        for state in self._states:
            c += len(self._transitions[state])
        return c

    def __iter__(self):
        """Iterates over (start, event, end) transition tuples."""
        for state in self._states:
            for event, target in self._transitions[state].items():
                yield (state, event, target.name)

    def pformat(self, sort=True, empty='.'):
        """Pretty formats the state + transition table into a string.

        NOTE(harlowja): the sort parameter can be provided to sort the states
        and transitions by sort order; with it being provided as false the rows
        will be iterated in addition order instead.
        """
        tbl = prettytable.PrettyTable(['Start', 'Event', 'End', 'On Enter', 'On Exit'])
        for state in _orderedkeys(self._states, sort=sort):
            prefix_markings = []
            if self.current_state == state:
                prefix_markings.append('@')
            postfix_markings = []
            if self.default_start_state == state:
                postfix_markings.append('^')
            if self._states[state]['terminal']:
                postfix_markings.append('$')
            pretty_state = '%s%s' % (''.join(prefix_markings), state)
            if postfix_markings:
                pretty_state += '[%s]' % ''.join(postfix_markings)
            if self._transitions[state]:
                for event in _orderedkeys(self._transitions[state], sort=sort):
                    target = self._transitions[state][event]
                    row = [pretty_state, event, target.name]
                    if target.on_enter is not None:
                        row.append(utils.get_callback_name(target.on_enter))
                    else:
                        row.append(empty)
                    if target.on_exit is not None:
                        row.append(utils.get_callback_name(target.on_exit))
                    else:
                        row.append(empty)
                    tbl.add_row(row)
            else:
                on_enter = self._states[state]['on_enter']
                if on_enter is not None:
                    on_enter = utils.get_callback_name(on_enter)
                else:
                    on_enter = empty
                on_exit = self._states[state]['on_exit']
                if on_exit is not None:
                    on_exit = utils.get_callback_name(on_exit)
                else:
                    on_exit = empty
                tbl.add_row([pretty_state, empty, empty, on_enter, on_exit])
        return tbl.get_string()