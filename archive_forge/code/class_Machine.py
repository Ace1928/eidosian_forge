import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
class Machine(object):
    """ Machine manages states, transitions and models. In case it is initialized without a specific model
    (or specifically no model), it will also act as a model itself. Machine takes also care of decorating
    models with conveniences functions related to added transitions and states during runtime.

    Attributes:
        states (OrderedDict): Collection of all registered states.
        events (dict): Collection of transitions ordered by trigger/event.
        models (list): List of models attached to the machine.
        initial (str): Name of the initial state for new models.
        prepare_event (list): Callbacks executed when an event is triggered.
        before_state_change (list): Callbacks executed after condition checks but before transition is conducted.
            Callbacks will be executed BEFORE the custom callbacks assigned to the transition.
        after_state_change (list): Callbacks executed after the transition has been conducted.
            Callbacks will be executed AFTER the custom callbacks assigned to the transition.
        finalize_event (list): Callbacks will be executed after all transitions callbacks have been executed.
            Callbacks mentioned here will also be called if a transition or condition check raised an error.
        queued (bool): Whether transitions in callbacks should be executed immediately (False) or sequentially.
        send_event (bool): When True, any arguments passed to trigger methods will be wrapped in an EventData
            object, allowing indirect and encapsulated access to data. When False, all positional and keyword
            arguments will be passed directly to all callback methods.
        auto_transitions (bool):  When True (default), every state will automatically have an associated
            to_{state}() convenience trigger in the base model.
        ignore_invalid_triggers (bool): When True, any calls to trigger methods that are not valid for the
            present state (e.g., calling an a_to_b() trigger when the current state is c) will be silently
            ignored rather than raising an invalid transition exception.
        name (str): Name of the ``Machine`` instance mainly used for easier log message distinction.
    """
    separator = '_'
    wildcard_all = '*'
    wildcard_same = '='
    state_cls = State
    transition_cls = Transition
    event_cls = Event
    self_literal = 'self'

    def __init__(self, model=self_literal, states=None, initial='initial', transitions=None, send_event=False, auto_transitions=True, ordered_transitions=False, ignore_invalid_triggers=None, before_state_change=None, after_state_change=None, name=None, queued=False, prepare_event=None, finalize_event=None, model_attribute='state', on_exception=None, **kwargs):
        """
        Args:
            model (object or list): The object(s) whose states we want to manage. If set to `Machine.self_literal`
                (default value), the current Machine instance will be used as the model (i.e., all
                triggering events will be attached to the Machine itself). Note that an empty list
                is treated like no model.
            states (list or Enum): A list or enumeration of valid states. Each list element can be either a
                string, an enum member or a State instance. If string or enum member, a new generic State
                instance will be created that is named according to the string or enum member's name.
            initial (str, Enum or State): The initial state of the passed model[s].
            transitions (list): An optional list of transitions. Each element
                is a dictionary of named arguments to be passed onto the
                Transition initializer.
            send_event (boolean): When True, any arguments passed to trigger
                methods will be wrapped in an EventData object, allowing
                indirect and encapsulated access to data. When False, all
                positional and keyword arguments will be passed directly to all
                callback methods.
            auto_transitions (boolean): When True (default), every state will
                automatically have an associated to_{state}() convenience
                trigger in the base model.
            ordered_transitions (boolean): Convenience argument that calls
                add_ordered_transitions() at the end of initialization if set
                to True.
            ignore_invalid_triggers: when True, any calls to trigger methods
                that are not valid for the present state (e.g., calling an
                a_to_b() trigger when the current state is c) will be silently
                ignored rather than raising an invalid transition exception.
            before_state_change: A callable called on every change state before
                the transition happened. It receives the very same args as normal
                callbacks.
            after_state_change: A callable called on every change state after
                the transition happened. It receives the very same args as normal
                callbacks.
            name: If a name is set, it will be used as a prefix for logger output
            queued (boolean): When True, processes transitions sequentially. A trigger
                executed in a state callback function will be queued and executed later.
                Due to the nature of the queued processing, all transitions will
                _always_ return True since conditional checks cannot be conducted at queueing time.
            prepare_event: A callable called on for before possible transitions will be processed.
                It receives the very same args as normal callbacks.
            finalize_event: A callable called on for each triggered event after transitions have been processed.
                This is also called when a transition raises an exception.
            on_exception: A callable called when an event raises an exception. If not set,
                the exception will be raised instead.

            **kwargs additional arguments passed to next class in MRO. This can be ignored in most cases.
        """
        try:
            super(Machine, self).__init__(**kwargs)
        except TypeError as err:
            raise ValueError('Passing arguments {0} caused an inheritance error: {1}'.format(kwargs.keys(), err))
        self._queued = queued
        self._transition_queue = deque()
        self._before_state_change = []
        self._after_state_change = []
        self._prepare_event = []
        self._finalize_event = []
        self._on_exception = []
        self._initial = None
        self.states = OrderedDict()
        self.events = {}
        self.send_event = send_event
        self.auto_transitions = auto_transitions
        self.ignore_invalid_triggers = ignore_invalid_triggers
        self.prepare_event = prepare_event
        self.before_state_change = before_state_change
        self.after_state_change = after_state_change
        self.finalize_event = finalize_event
        self.on_exception = on_exception
        self.name = name + ': ' if name is not None else ''
        self.model_attribute = model_attribute
        self.models = []
        if states is not None:
            self.add_states(states)
        if initial is not None:
            self.initial = initial
        if transitions is not None:
            self.add_transitions(transitions)
        if ordered_transitions:
            self.add_ordered_transitions()
        if model:
            self.add_model(model)

    def add_model(self, model, initial=None):
        """ Register a model with the state machine, initializing triggers and callbacks. """
        models = listify(model)
        if initial is None:
            if self.initial is None:
                raise ValueError('No initial state configured for machine, must specify when adding model.')
            initial = self.initial
        for mod in models:
            mod = self if mod is self.self_literal else mod
            if mod not in self.models:
                self._checked_assignment(mod, 'trigger', partial(self._get_trigger, mod))
                for trigger in self.events:
                    self._add_trigger_to_model(trigger, mod)
                for state in self.states.values():
                    self._add_model_to_state(state, mod)
                self.set_state(initial, model=mod)
                self.models.append(mod)

    def remove_model(self, model):
        """ Remove a model from the state machine. The model will still contain all previously added triggers
        and callbacks, but will not receive updates when states or transitions are added to the Machine.
        If an event queue is used, all queued events of that model will be removed."""
        models = listify(model)
        for mod in models:
            self.models.remove(mod)
        if len(self._transition_queue) > 0:
            self._transition_queue = deque([self._transition_queue[0]] + [e for e in self._transition_queue if e.args[0].model not in models])

    @classmethod
    def _create_transition(cls, *args, **kwargs):
        return cls.transition_cls(*args, **kwargs)

    @classmethod
    def _create_event(cls, *args, **kwargs):
        return cls.event_cls(*args, **kwargs)

    @classmethod
    def _create_state(cls, *args, **kwargs):
        return cls.state_cls(*args, **kwargs)

    @property
    def initial(self):
        """ Return the initial state. """
        return self._initial

    @initial.setter
    def initial(self, value):
        if isinstance(value, State):
            if value.name not in self.states:
                self.add_states(value)
            else:
                _ = self._has_state(value, raise_error=True)
            self._initial = value.name
        else:
            state_name = value.name if isinstance(value, Enum) else value
            if state_name not in self.states:
                self.add_states(state_name)
            self._initial = state_name

    @property
    def has_queue(self):
        """ Return boolean indicating if machine has queue or not """
        return self._queued

    @property
    def model(self):
        """ List of models attached to the machine. For backwards compatibility, the property will
        return the model instance itself instead of the underlying list  if there is only one attached
        to the machine.
        """
        if len(self.models) == 1:
            return self.models[0]
        return self.models

    @property
    def before_state_change(self):
        """Callbacks executed after condition checks but before transition is conducted.
        Callbacks will be executed BEFORE the custom callbacks assigned to the transition."""
        return self._before_state_change

    @before_state_change.setter
    def before_state_change(self, value):
        self._before_state_change = listify(value)

    @property
    def after_state_change(self):
        """Callbacks executed after the transition has been conducted.
        Callbacks will be executed AFTER the custom callbacks assigned to the transition."""
        return self._after_state_change

    @after_state_change.setter
    def after_state_change(self, value):
        self._after_state_change = listify(value)

    @property
    def prepare_event(self):
        """Callbacks executed when an event is triggered."""
        return self._prepare_event

    @prepare_event.setter
    def prepare_event(self, value):
        self._prepare_event = listify(value)

    @property
    def finalize_event(self):
        """Callbacks will be executed after all transitions callbacks have been executed.
        Callbacks mentioned here will also be called if a transition or condition check raised an error."""
        return self._finalize_event

    @finalize_event.setter
    def finalize_event(self, value):
        self._finalize_event = listify(value)

    @property
    def on_exception(self):
        """Callbacks will be executed when an Event raises an Exception."""
        return self._on_exception

    @on_exception.setter
    def on_exception(self, value):
        self._on_exception = listify(value)

    def get_state(self, state):
        """ Return the State instance with the passed name. """
        if isinstance(state, Enum):
            state = state.name
        if state not in self.states:
            raise ValueError("State '%s' is not a registered state." % state)
        return self.states[state]

    def is_state(self, state, model):
        """ Check whether the current state matches the named state. This function is not called directly
            but assigned as partials to model instances (e.g. is_A -> partial(_is_state, 'A', model)).
        Args:
            state (str or Enum): name of the checked state or Enum
            model: model to be checked
        Returns:
            bool: Whether the model's current state is state.
        """
        return getattr(model, self.model_attribute) == state

    def get_model_state(self, model):
        """
            Get the state of a model
        Args:
            model (object): the stateful model
        Returns:
            State: The State object related to the model's state
        """
        return self.get_state(getattr(model, self.model_attribute))

    def set_state(self, state, model=None):
        """
            Set the current state.
        Args:
            state (str or Enum or State): value of state to be set
            model (optional[object]): targeted model; if not set, all models will be set to 'state'
        """
        if not isinstance(state, State):
            state = self.get_state(state)
        models = self.models if model is None else listify(model)
        for mod in models:
            setattr(mod, self.model_attribute, state.value)

    def add_state(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
        """ Alias for add_states. """
        self.add_states(states=states, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)

    def add_states(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
        """ Add new state(s).
        Args:
            states (list, str, dict, Enum or State): a list, a State instance, the
                name of a new state, an enumeration (member) or a dict with keywords to pass on to the
                State initializer. If a list, each element can be a string, State or enumeration member.
            on_enter (str or list): callbacks to trigger when the state is
                entered. Only valid if first argument is string.
            on_exit (str or list): callbacks to trigger when the state is
                exited. Only valid if first argument is string.
            ignore_invalid_triggers: when True, any calls to trigger methods
                that are not valid for the present state (e.g., calling an
                a_to_b() trigger when the current state is c) will be silently
                ignored rather than raising an invalid transition exception.
                Note that this argument takes precedence over the same
                argument defined at the Machine level, and is in turn
                overridden by any ignore_invalid_triggers explicitly
                passed in an individual state's initialization arguments.

            **kwargs additional keyword arguments used by state mixins.
        """
        ignore = ignore_invalid_triggers
        if ignore is None:
            ignore = self.ignore_invalid_triggers
        states = listify(states)
        for state in states:
            if isinstance(state, (string_types, Enum)):
                state = self._create_state(state, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore, **kwargs)
            elif isinstance(state, dict):
                if 'ignore_invalid_triggers' not in state:
                    state['ignore_invalid_triggers'] = ignore
                state = self._create_state(**state)
            self.states[state.name] = state
            for model in self.models:
                self._add_model_to_state(state, model)
            if self.auto_transitions:
                for a_state in self.states.keys():
                    if a_state == state.name:
                        if self.model_attribute == 'state':
                            method_name = 'to_%s' % a_state
                        else:
                            method_name = 'to_%s_%s' % (self.model_attribute, a_state)
                        self.add_transition(method_name, self.wildcard_all, a_state)
                    else:
                        if self.model_attribute == 'state':
                            method_name = 'to_%s' % a_state
                        else:
                            method_name = 'to_%s_%s' % (self.model_attribute, a_state)
                        self.add_transition(method_name, state.name, a_state)

    def _add_model_to_state(self, state, model):
        func = partial(self.is_state, state.value, model)
        if self.model_attribute == 'state':
            method_name = 'is_%s' % state.name
        else:
            method_name = 'is_%s_%s' % (self.model_attribute, state.name)
        self._checked_assignment(model, method_name, func)
        for callback in self.state_cls.dynamic_methods:
            method = '{0}_{1}'.format(callback, state.name)
            if hasattr(model, method) and inspect.ismethod(getattr(model, method)) and (method not in getattr(state, callback)):
                state.add_callback(callback[3:], method)

    def _checked_assignment(self, model, name, func):
        if hasattr(model, name):
            _LOGGER.warning("%sModel already contains an attribute '%s'. Skip binding.", self.name, name)
        else:
            setattr(model, name, func)

    def _can_trigger(self, model, trigger, *args, **kwargs):
        evt = EventData(None, None, self, model, args, kwargs)
        state = self.get_model_state(model).name
        for trigger_name in self.get_triggers(state):
            if trigger_name != trigger:
                continue
            for transition in self.events[trigger_name].transitions[state]:
                try:
                    _ = self.get_state(transition.dest)
                except ValueError:
                    continue
                self.callbacks(self.prepare_event, evt)
                self.callbacks(transition.prepare, evt)
                if all((c.check(evt) for c in transition.conditions)):
                    return True
        return False

    def _add_may_transition_func_for_trigger(self, trigger, model):
        self._checked_assignment(model, 'may_%s' % trigger, partial(self._can_trigger, model, trigger))

    def _add_trigger_to_model(self, trigger, model):
        self._checked_assignment(model, trigger, partial(self.events[trigger].trigger, model))
        self._add_may_transition_func_for_trigger(trigger, model)

    def _get_trigger(self, model, trigger_name, *args, **kwargs):
        """Convenience function added to the model to trigger events by name.
        Args:
            model (object): Model with assigned event trigger.
            trigger_name (str): Name of the trigger to be called.
            *args: Variable length argument list which is passed to the triggered event.
            **kwargs: Arbitrary keyword arguments which is passed to the triggered event.
        Returns:
            bool: True if a transitions has been conducted or the trigger event has been queued.
        """
        try:
            event = self.events[trigger_name]
        except KeyError:
            state = self.get_model_state(model)
            ignore = state.ignore_invalid_triggers if state.ignore_invalid_triggers is not None else self.ignore_invalid_triggers
            if not ignore:
                raise AttributeError("Do not know event named '%s'." % trigger_name)
            return False
        return event.trigger(model, *args, **kwargs)

    def get_triggers(self, *args):
        """ Collects all triggers FROM certain states.
        Args:
            *args: Tuple of source states.

        Returns:
            list of transition/trigger names.
        """
        names = {state.name if hasattr(state, 'name') else state for state in args}
        return [t for t, ev in self.events.items() if any((name in ev.transitions for name in names))]

    def add_transition(self, trigger, source, dest, conditions=None, unless=None, before=None, after=None, prepare=None, **kwargs):
        """ Create a new Transition instance and add it to the internal list.
        Args:
            trigger (str): The name of the method that will trigger the
                transition. This will be attached to the currently specified
                model (e.g., passing trigger='advance' will create a new
                advance() method in the model that triggers the transition.)
            source(str, Enum or list): The name of the source state--i.e., the state we
                are transitioning away from. This can be a single state, a
                list of states or an asterisk for all states.
            dest (str or Enum): The name of the destination State--i.e., the state
                we are transitioning into. This can be a single state or an
                equal sign to specify that the transition should be reflexive
                so that the destination will be the same as the source for
                every given source. If dest is None, this transition will be
                an internal transition (exit/enter callbacks won't be processed).
            conditions (str or list): Condition(s) that must pass in order
                for the transition to take place. Either a list providing the
                name of a callable, or a list of callables. For the transition
                to occur, ALL callables must return True.
            unless (str or list): Condition(s) that must return False in order
                for the transition to occur. Behaves just like conditions arg
                otherwise.
            before (str or list): Callables to call before the transition.
            after (str or list): Callables to call after the transition.
            prepare (str or list): Callables to call when the trigger is activated
            **kwargs: Additional arguments which can be passed to the created transition.
                This is useful if you plan to extend Machine.Transition and require more parameters.
        """
        if trigger == self.model_attribute:
            raise ValueError('Trigger name cannot be same as model attribute name.')
        if trigger not in self.events:
            self.events[trigger] = self._create_event(trigger, self)
            for model in self.models:
                self._add_trigger_to_model(trigger, model)
        if source == self.wildcard_all:
            source = list(self.states.keys())
        else:
            source = [s.name if isinstance(s, State) and self._has_state(s, raise_error=True) or hasattr(s, 'name') else s for s in listify(source)]
        for state in source:
            if dest == self.wildcard_same:
                _dest = state
            elif dest is not None:
                if isinstance(dest, State):
                    _ = self._has_state(dest, raise_error=True)
                _dest = dest.name if hasattr(dest, 'name') else dest
            else:
                _dest = None
            _trans = self._create_transition(state, _dest, conditions, unless, before, after, prepare, **kwargs)
            self.events[trigger].add_transition(_trans)

    def add_transitions(self, transitions):
        """ Add several transitions.

        Args:
            transitions (list): A list of transitions.

        """
        for trans in listify(transitions):
            if isinstance(trans, list):
                self.add_transition(*trans)
            else:
                self.add_transition(**trans)

    def add_ordered_transitions(self, states=None, trigger='next_state', loop=True, loop_includes_initial=True, conditions=None, unless=None, before=None, after=None, prepare=None, **kwargs):
        """ Add a set of transitions that move linearly from state to state.
        Args:
            states (list): A list of state names defining the order of the
                transitions. E.g., ['A', 'B', 'C'] will generate transitions
                for A --> B, B --> C, and C --> A (if loop is True). If states
                is None, all states in the current instance will be used.
            trigger (str): The name of the trigger method that advances to
                the next state in the sequence.
            loop (boolean): Whether to add a transition from the last
                state to the first state.
            loop_includes_initial (boolean): If no initial state was defined in
                the machine, setting this to True will cause the _initial state
                placeholder to be included in the added transitions. This argument
                has no effect if the states argument is passed without the
                initial state included.
            conditions (str or list): Condition(s) that must pass in order
                for the transition to take place. Either a list providing the
                name of a callable, or a list of callables. For the transition
                to occur, ALL callables must return True.
            unless (str or list): Condition(s) that must return False in order
                for the transition to occur. Behaves just like conditions arg
                otherwise.
            before (str or list): Callables to call before the transition.
            after (str or list): Callables to call after the transition.
            prepare (str or list): Callables to call when the trigger is activated
            **kwargs: Additional arguments which can be passed to the created transition.
                This is useful if you plan to extend Machine.Transition and require more parameters.
        """
        if states is None:
            states = list(self.states.keys())
        len_transitions = len(states)
        if len_transitions < 2:
            raise ValueError("Can't create ordered transitions on a Machine with fewer than 2 states.")
        if not loop:
            len_transitions -= 1
        conditions = _prep_ordered_arg(len_transitions, conditions)
        unless = _prep_ordered_arg(len_transitions, unless)
        before = _prep_ordered_arg(len_transitions, before)
        after = _prep_ordered_arg(len_transitions, after)
        prepare = _prep_ordered_arg(len_transitions, prepare)
        try:
            idx = states.index(self._initial)
            states = states[idx:] + states[:idx]
            first_in_loop = states[0 if loop_includes_initial else 1]
        except ValueError:
            first_in_loop = states[0]
        for i in range(0, len(states) - 1):
            self.add_transition(trigger, states[i], states[i + 1], conditions=conditions[i], unless=unless[i], before=before[i], after=after[i], prepare=prepare[i], **kwargs)
        if loop:
            self.add_transition(trigger, states[-1], first_in_loop, conditions=conditions[-1], unless=unless[-1], before=before[-1], after=after[-1], prepare=prepare[-1], **kwargs)

    def get_transitions(self, trigger='', source='*', dest='*'):
        """ Return the transitions from the Machine.
        Args:
            trigger (str): Trigger name of the transition.
            source (str, Enum or State): Limits list to transitions from a certain state.
            dest (str, Enum or State): Limits list to transitions to a certain state.
        """
        if trigger:
            try:
                events = (self.events[trigger],)
            except KeyError:
                return []
        else:
            events = self.events.values()
        transitions = []
        for event in events:
            transitions.extend(itertools.chain.from_iterable(event.transitions.values()))
        target_source = source.name if hasattr(source, 'name') else source if source != '*' else ''
        target_dest = dest.name if hasattr(dest, 'name') else dest if dest != '*' else ''
        return [transition for transition in transitions if (transition.source, transition.dest) == (target_source or transition.source, target_dest or transition.dest)]

    def remove_transition(self, trigger, source='*', dest='*'):
        """ Removes a transition from the Machine and all models.
        Args:
            trigger (str): Trigger name of the transition.
            source (str, Enum or State): Limits removal to transitions from a certain state.
            dest (str, Enum or State): Limits removal to transitions to a certain state.
        """
        source = listify(source) if source != '*' else source
        dest = listify(dest) if dest != '*' else dest
        tmp = {key: value for key, value in {k: [t for t in v if source != '*' and t.source not in source or (dest != '*' and t.dest not in dest)] for k, v in self.events[trigger].transitions.items()}.items() if len(value) > 0}
        if tmp:
            self.events[trigger].transitions = defaultdict(list, **tmp)
        else:
            for model in self.models:
                delattr(model, trigger)
            del self.events[trigger]

    def dispatch(self, trigger, *args, **kwargs):
        """ Trigger an event on all models assigned to the machine.
        Args:
            trigger (str): Event name
            *args (list): List of arguments passed to the event trigger
            **kwargs (dict): Dictionary of keyword arguments passed to the event trigger
        Returns:
            bool The truth value of all triggers combined with AND
        """
        return all((getattr(model, trigger)(*args, **kwargs) for model in self.models))

    def callbacks(self, funcs, event_data):
        """ Triggers a list of callbacks """
        for func in funcs:
            self.callback(func, event_data)
            _LOGGER.info("%sExecuted callback '%s'", self.name, func)

    def callback(self, func, event_data):
        """ Trigger a callback function with passed event_data parameters. In case func is a string,
            the callable will be resolved from the passed model in event_data. This function is not intended to
            be called directly but through state and transition callback definitions.
        Args:
            func (str or callable): The callback function.
                1. First, if the func is callable, just call it
                2. Second, we try to import string assuming it is a path to a func
                3. Fallback to a model attribute
            event_data (EventData): An EventData instance to pass to the
                callback (if event sending is enabled) or to extract arguments
                from (if event sending is disabled).
        """
        func = self.resolve_callable(func, event_data)
        if self.send_event:
            func(event_data)
        else:
            func(*event_data.args, **event_data.kwargs)

    @staticmethod
    def resolve_callable(func, event_data):
        """ Converts a model's property name, method name or a path to a callable into a callable.
            If func is not a string it will be returned unaltered.
        Args:
            func (str or callable): Property name, method name or a path to a callable
            event_data (EventData): Currently processed event
        Returns:
            callable function resolved from string or func
        """
        if isinstance(func, string_types):
            try:
                func = getattr(event_data.model, func)
                if not callable(func):

                    def func_wrapper(*_, **__):
                        return func
                    return func_wrapper
            except AttributeError:
                try:
                    module_name, func_name = func.rsplit('.', 1)
                    module = __import__(module_name)
                    for submodule_name in module_name.split('.')[1:]:
                        module = getattr(module, submodule_name)
                    func = getattr(module, func_name)
                except (ImportError, AttributeError, ValueError):
                    raise AttributeError("Callable with name '%s' could neither be retrieved from the passed model nor imported from a module." % func)
        return func

    def _has_state(self, state, raise_error=False):
        found = state in self.states.values()
        if not found and raise_error:
            msg = 'State %s has not been added to the machine' % (state.name if hasattr(state, 'name') else state)
            raise ValueError(msg)
        return found

    def _process(self, trigger):
        if not self.has_queue:
            if not self._transition_queue:
                return trigger()
            raise MachineError('Attempt to process events synchronously while transition queue is not empty!')
        self._transition_queue.append(trigger)
        if len(self._transition_queue) > 1:
            return True
        while self._transition_queue:
            try:
                self._transition_queue[0]()
                self._transition_queue.popleft()
            except Exception:
                self._transition_queue.clear()
                raise
        return True

    def _identify_callback(self, name):
        for callback in itertools.chain(self.state_cls.dynamic_methods, self.transition_cls.dynamic_methods):
            if name.startswith(callback):
                callback_type = callback
                break
        else:
            return (None, None)
        target = name[len(callback_type) + len(self.separator):]
        if target == '' or name[len(callback_type)] != self.separator:
            return (None, None)
        return (callback_type, target)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError("'{}' does not exist on <Machine@{}>".format(name, id(self)))
        callback_type, target = self._identify_callback(name)
        if callback_type is not None:
            if callback_type in self.transition_cls.dynamic_methods:
                if target not in self.events:
                    raise AttributeError("event '{}' is not registered on <Machine@{}>".format(target, id(self)))
                return partial(self.events[target].add_callback, callback_type)
            if callback_type in self.state_cls.dynamic_methods:
                state = self.get_state(target)
                return partial(state.add_callback, callback_type[3:])
        try:
            return self.__getattribute__(name)
        except AttributeError:
            raise AttributeError("'{}' does not exist on <Machine@{}>".format(name, id(self)))