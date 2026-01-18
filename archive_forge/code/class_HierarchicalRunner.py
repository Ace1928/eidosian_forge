import abc
from automaton import exceptions as excp
from automaton import machines
class HierarchicalRunner(Runner):
    """Hierarchical machine runner used to run a hierarchical machine.

    Only **one** runner per machine should be active at the same time (aka
    there should not be multiple runners using the same machine instance at
    the same time).
    """

    def __init__(self, machine):
        """Create a runner for the given machine."""
        if not isinstance(machine, (machines.HierarchicalFiniteMachine,)):
            raise TypeError('HierarchicalRunner only works with HierarchicalFiniteMachine(s)')
        super(HierarchicalRunner, self).__init__(machine)

    def run(self, event, initialize=True):
        for transition in self.run_iter(event, initialize=initialize):
            pass

    @staticmethod
    def _process_event(machines, event):
        """Matches a event to the machine hierarchy.

        If the lowest level machine does not handle the event, then the
        parent machine is referred to and so on, until there is only one
        machine left which *must* handle the event.

        The machine whose ``process_event`` does not throw invalid state or
        not found exceptions is expected to be the machine that should
        continue handling events...
        """
        while True:
            machine = machines[-1]
            try:
                result = machine.process_event(event)
            except (excp.InvalidState, excp.NotFound):
                if len(machines) == 1:
                    raise
                else:
                    current = machine._current
                    if current is not None and current.on_exit is not None:
                        current.on_exit(current.name, event)
                    machine._current = None
                    machines.pop()
            else:
                return result

    def run_iter(self, event, initialize=True):
        """Returns a iterator/generator that will run the state machine.

        This will keep a stack (hierarchy) of machines active and jumps through
        them as needed (depending on which machine handles which event) during
        the running lifecycle.

        NOTE(harlowja): only one runner iterator/generator should be active for
        a machine hierarchy, if this is not observed then it is possible for
        initialization and other local state to be corrupted and causes issues
        when running...
        """
        machines = [self._machine]
        if initialize:
            machines[-1].initialize()
        while True:
            old_state = machines[-1].current_state
            effect = self._process_event(machines, event)
            new_state = machines[-1].current_state
            try:
                machine = effect.machine
            except AttributeError:
                pass
            else:
                if machine is not None and machine is not machines[-1]:
                    machine.initialize()
                    machines.append(machine)
            try:
                sent_event = (yield (old_state, new_state))
            except GeneratorExit:
                break
            if len(machines) == 1 and effect.terminal:
                break
            if effect.reaction is None and sent_event is None:
                raise excp.NotFound(_JUMPER_NOT_FOUND_TPL % (new_state, old_state, event))
            elif sent_event is not None:
                event = sent_event
            else:
                cb, args, kwargs = effect.reaction
                event = cb(old_state, new_state, event, *args, **kwargs)