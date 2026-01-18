import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
class HierarchicalFiniteMachine(FiniteMachine):
    """A fsm that understands how to run in a hierarchical mode."""
    Effect = collections.namedtuple('Effect', 'reaction,terminal,machine')

    def __init__(self):
        super(HierarchicalFiniteMachine, self).__init__()
        self._nested_machines = {}

    @classmethod
    def _effect_builder(cls, new_state, event):
        return cls.Effect(new_state['reactions'].get(event), new_state['terminal'], new_state.get('machine'))

    def add_state(self, state, terminal=False, on_enter=None, on_exit=None, machine=None):
        """Adds a given state to the state machine.

        :param machine: the nested state machine that will be transitioned
                        into when this state is entered
        :type machine: :py:class:`.FiniteMachine`

        Further arguments are interpreted as
        for :py:meth:`.FiniteMachine.add_state`.
        """
        if machine is not None and (not isinstance(machine, FiniteMachine)):
            raise ValueError('Nested state machines must themselves be state machines')
        super(HierarchicalFiniteMachine, self).add_state(state, terminal=terminal, on_enter=on_enter, on_exit=on_exit)
        if machine is not None:
            self._states[state]['machine'] = machine
            self._nested_machines[state] = machine

    def copy(self, shallow=False, unfreeze=False):
        c = super(HierarchicalFiniteMachine, self).copy(shallow=shallow, unfreeze=unfreeze)
        if shallow:
            c._nested_machines = self._nested_machines
        else:
            c._nested_machines = self._nested_machines.copy()
        return c

    def initialize(self, start_state=None, nested_start_state_fetcher=None):
        """Sets up the state machine (sets current state to start state...).

        :param start_state: explicit start state to use to initialize the
                            state machine to. If ``None`` is provided then the
                            machine's default start state will be used
                            instead.
        :param nested_start_state_fetcher: A callback that can return start
                                           states for any nested machines
                                           **only**. If not ``None`` then it
                                           will be provided a single argument,
                                           the machine to provide a starting
                                           state for and it is expected to
                                           return a starting state (or
                                           ``None``) for each machine called
                                           with. Do note that this callback
                                           will also be passed to other nested
                                           state machines as well, so it will
                                           also be used to initialize any state
                                           machines they contain (recursively).
        """
        super(HierarchicalFiniteMachine, self).initialize(start_state=start_state)
        for data in self._states.values():
            if 'machine' in data:
                nested_machine = data['machine']
                nested_start_state = None
                if nested_start_state_fetcher is not None:
                    nested_start_state = nested_start_state_fetcher(nested_machine)
                if isinstance(nested_machine, HierarchicalFiniteMachine):
                    nested_machine.initialize(start_state=nested_start_state, nested_start_state_fetcher=nested_start_state_fetcher)
                else:
                    nested_machine.initialize(start_state=nested_start_state)

    @property
    def nested_machines(self):
        """Dictionary of **all** nested state machines this machine may use."""
        return self._nested_machines