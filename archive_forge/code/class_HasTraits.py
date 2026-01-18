from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class HasTraits(HasDescriptors, metaclass=MetaHasTraits):
    _trait_values: dict[str, t.Any]
    _static_immutable_initial_values: dict[str, t.Any]
    _trait_notifiers: dict[str | Sentinel, t.Any]
    _trait_validators: dict[str | Sentinel, t.Any]
    _cross_validation_lock: bool
    _traits: dict[str, t.Any]
    _all_trait_default_generators: dict[str, t.Any]

    def setup_instance(*args: t.Any, **kwargs: t.Any) -> None:
        self = args[0]
        args = args[1:]
        self._trait_values = self._static_immutable_initial_values.copy()
        self._trait_notifiers = {}
        self._trait_validators = {}
        self._cross_validation_lock = False
        super(HasTraits, self).setup_instance(*args, **kwargs)

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super_args = args
        super_kwargs = {}
        if kwargs:

            def ignore(change: Bunch) -> None:
                pass
            self.notify_change = ignore
            self._cross_validation_lock = True
            changes = {}
            for key, value in kwargs.items():
                if self.has_trait(key):
                    setattr(self, key, value)
                    changes[key] = Bunch(name=key, old=None, new=value, owner=self, type='change')
                else:
                    super_kwargs[key] = value
            changed = set(kwargs) & set(self._traits)
            for key in changed:
                value = self._traits[key]._cross_validate(self, getattr(self, key))
                self.set_trait(key, value)
                changes[key]['new'] = value
            self._cross_validation_lock = False
            del self.notify_change
            for key in changed:
                self.notify_change(changes[key])
        try:
            super().__init__(*super_args, **super_kwargs)
        except TypeError as e:
            arg_s_list = [repr(arg) for arg in super_args]
            for k, v in super_kwargs.items():
                arg_s_list.append(f'{k}={v!r}')
            arg_s = ', '.join(arg_s_list)
            warn('Passing unrecognized arguments to super({classname}).__init__({arg_s}).\n{error}\nThis is deprecated in traitlets 4.2.This error will be raised in a future release of traitlets.'.format(arg_s=arg_s, classname=self.__class__.__name__, error=e), DeprecationWarning, stacklevel=2)

    def __getstate__(self) -> dict[str, t.Any]:
        d = self.__dict__.copy()
        d['_trait_notifiers'] = {}
        d['_trait_validators'] = {}
        d['_trait_values'] = self._trait_values.copy()
        d['_cross_validation_lock'] = False
        return d

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self.__dict__ = state.copy()
        cls = self.__class__
        for key in dir(cls):
            try:
                value = getattr(cls, key)
            except AttributeError:
                pass
            else:
                if isinstance(value, EventHandler):
                    value.instance_init(self)

    @property
    @contextlib.contextmanager
    def cross_validation_lock(self) -> t.Any:
        """
        A contextmanager for running a block with our cross validation lock set
        to True.

        At the end of the block, the lock's value is restored to its value
        prior to entering the block.
        """
        if self._cross_validation_lock:
            yield
            return
        else:
            try:
                self._cross_validation_lock = True
                yield
            finally:
                self._cross_validation_lock = False

    @contextlib.contextmanager
    def hold_trait_notifications(self) -> t.Any:
        """Context manager for bundling trait change notifications and cross
        validation.

        Use this when doing multiple trait assignments (init, config), to avoid
        race conditions in trait notifiers requesting other trait values.
        All trait notifications will fire after all values have been assigned.
        """
        if self._cross_validation_lock:
            yield
            return
        else:
            cache: dict[str, list[Bunch]] = {}

            def compress(past_changes: list[Bunch] | None, change: Bunch) -> list[Bunch]:
                """Merges the provided change with the last if possible."""
                if past_changes is None:
                    return [change]
                else:
                    if past_changes[-1]['type'] == 'change' and change.type == 'change':
                        past_changes[-1]['new'] = change.new
                    else:
                        past_changes.append(change)
                    return past_changes

            def hold(change: Bunch) -> None:
                name = change.name
                cache[name] = compress(cache.get(name), change)
            try:
                self.notify_change = hold
                self._cross_validation_lock = True
                yield
                for name in list(cache.keys()):
                    trait = getattr(self.__class__, name)
                    value = trait._cross_validate(self, getattr(self, name))
                    self.set_trait(name, value)
            except TraitError as e:
                self.notify_change = lambda x: None
                for name, changes in cache.items():
                    for change in changes[::-1]:
                        if change.type == 'change':
                            if change.old is not Undefined:
                                self.set_trait(name, change.old)
                            else:
                                self._trait_values.pop(name)
                cache = {}
                raise e
            finally:
                self._cross_validation_lock = False
                del self.notify_change
                for changes in cache.values():
                    for change in changes:
                        self.notify_change(change)

    def _notify_trait(self, name: str, old_value: t.Any, new_value: t.Any) -> None:
        self.notify_change(Bunch(name=name, old=old_value, new=new_value, owner=self, type='change'))

    def notify_change(self, change: Bunch) -> None:
        """Notify observers of a change event"""
        return self._notify_observers(change)

    def _notify_observers(self, event: Bunch) -> None:
        """Notify observers of any event"""
        if not isinstance(event, Bunch):
            event = Bunch(event)
        name, type = (event['name'], event['type'])
        callables = []
        if name in self._trait_notifiers:
            callables.extend(self._trait_notifiers.get(name, {}).get(type, []))
            callables.extend(self._trait_notifiers.get(name, {}).get(All, []))
        if All in self._trait_notifiers:
            callables.extend(self._trait_notifiers.get(All, {}).get(type, []))
            callables.extend(self._trait_notifiers.get(All, {}).get(All, []))
        magic_name = '_%s_changed' % name
        if event['type'] == 'change' and hasattr(self, magic_name):
            class_value = getattr(self.__class__, magic_name)
            if not isinstance(class_value, ObserveHandler):
                deprecated_method(class_value, self.__class__, magic_name, 'use @observe and @unobserve instead.')
                cb = getattr(self, magic_name)
                if cb not in callables:
                    callables.append(_callback_wrapper(cb))
        for c in callables:
            if isinstance(c, _CallbackWrapper):
                c = c.__call__
            elif isinstance(c, EventHandler) and c.name is not None:
                c = getattr(self, c.name)
            c(event)

    def _add_notifiers(self, handler: t.Callable[..., t.Any], name: Sentinel | str, type: str | Sentinel) -> None:
        if name not in self._trait_notifiers:
            nlist: list[t.Any] = []
            self._trait_notifiers[name] = {type: nlist}
        elif type not in self._trait_notifiers[name]:
            nlist = []
            self._trait_notifiers[name][type] = nlist
        else:
            nlist = self._trait_notifiers[name][type]
        if handler not in nlist:
            nlist.append(handler)

    def _remove_notifiers(self, handler: t.Callable[..., t.Any] | None, name: Sentinel | str, type: str | Sentinel) -> None:
        try:
            if handler is None:
                del self._trait_notifiers[name][type]
            else:
                self._trait_notifiers[name][type].remove(handler)
        except KeyError:
            pass

    def on_trait_change(self, handler: EventHandler | None=None, name: Sentinel | str | None=None, remove: bool=False) -> None:
        """DEPRECATED: Setup a handler to be called when a trait changes.

        This is used to setup dynamic notifications of trait changes.

        Static handlers can be created by creating methods on a HasTraits
        subclass with the naming convention '_[traitname]_changed'.  Thus,
        to create static handler for the trait 'a', create the method
        _a_changed(self, name, old, new) (fewer arguments can be used, see
        below).

        If `remove` is True and `handler` is not specified, all change
        handlers for the specified name are uninstalled.

        Parameters
        ----------
        handler : callable, None
            A callable that is called when a trait changes.  Its
            signature can be handler(), handler(name), handler(name, new),
            handler(name, old, new), or handler(name, old, new, self).
        name : list, str, None
            If None, the handler will apply to all traits.  If a list
            of str, handler will apply to all names in the list.  If a
            str, the handler will apply just to that name.
        remove : bool
            If False (the default), then install the handler.  If True
            then unintall it.
        """
        warn('on_trait_change is deprecated in traitlets 4.1: use observe instead', DeprecationWarning, stacklevel=2)
        if name is None:
            name = All
        if remove:
            self.unobserve(_callback_wrapper(handler), names=name)
        else:
            self.observe(_callback_wrapper(handler), names=name)

    def observe(self, handler: t.Callable[..., t.Any], names: Sentinel | str | t.Iterable[Sentinel | str]=All, type: Sentinel | str='change') -> None:
        """Setup a handler to be called when a trait changes.

        This is used to setup dynamic notifications of trait changes.

        Parameters
        ----------
        handler : callable
            A callable that is called when a trait changes. Its
            signature should be ``handler(change)``, where ``change`` is a
            dictionary. The change dictionary at least holds a 'type' key.
            * ``type``: the type of notification.
            Other keys may be passed depending on the value of 'type'. In the
            case where type is 'change', we also have the following keys:
            * ``owner`` : the HasTraits instance
            * ``old`` : the old value of the modified trait attribute
            * ``new`` : the new value of the modified trait attribute
            * ``name`` : the name of the modified trait attribute.
        names : list, str, All
            If names is All, the handler will apply to all traits.  If a list
            of str, handler will apply to all names in the list.  If a
            str, the handler will apply just to that name.
        type : str, All (default: 'change')
            The type of notification to filter by. If equal to All, then all
            notifications are passed to the observe handler.
        """
        for name in parse_notifier_name(names):
            self._add_notifiers(handler, name, type)

    def unobserve(self, handler: t.Callable[..., t.Any], names: Sentinel | str | t.Iterable[Sentinel | str]=All, type: Sentinel | str='change') -> None:
        """Remove a trait change handler.

        This is used to unregister handlers to trait change notifications.

        Parameters
        ----------
        handler : callable
            The callable called when a trait attribute changes.
        names : list, str, All (default: All)
            The names of the traits for which the specified handler should be
            uninstalled. If names is All, the specified handler is uninstalled
            from the list of notifiers corresponding to all changes.
        type : str or All (default: 'change')
            The type of notification to filter by. If All, the specified handler
            is uninstalled from the list of notifiers corresponding to all types.
        """
        for name in parse_notifier_name(names):
            self._remove_notifiers(handler, name, type)

    def unobserve_all(self, name: str | t.Any=All) -> None:
        """Remove trait change handlers of any type for the specified name.
        If name is not specified, removes all trait notifiers."""
        if name is All:
            self._trait_notifiers = {}
        else:
            try:
                del self._trait_notifiers[name]
            except KeyError:
                pass

    def _register_validator(self, handler: t.Callable[..., None], names: tuple[str | Sentinel, ...]) -> None:
        """Setup a handler to be called when a trait should be cross validated.

        This is used to setup dynamic notifications for cross-validation.

        If a validator is already registered for any of the provided names, a
        TraitError is raised and no new validator is registered.

        Parameters
        ----------
        handler : callable
            A callable that is called when the given trait is cross-validated.
            Its signature is handler(proposal), where proposal is a Bunch (dictionary with attribute access)
            with the following attributes/keys:
                * ``owner`` : the HasTraits instance
                * ``value`` : the proposed value for the modified trait attribute
                * ``trait`` : the TraitType instance associated with the attribute
        names : List of strings
            The names of the traits that should be cross-validated
        """
        for name in names:
            magic_name = '_%s_validate' % name
            if hasattr(self, magic_name):
                class_value = getattr(self.__class__, magic_name)
                if not isinstance(class_value, ValidateHandler):
                    deprecated_method(class_value, self.__class__, magic_name, 'use @validate decorator instead.')
        for name in names:
            self._trait_validators[name] = handler

    def add_traits(self, **traits: t.Any) -> None:
        """Dynamically add trait attributes to the HasTraits instance."""
        cls = self.__class__
        attrs = {'__module__': cls.__module__}
        if hasattr(cls, '__qualname__'):
            attrs['__qualname__'] = cls.__qualname__
        attrs.update(traits)
        self.__class__ = type(cls.__name__, (cls,), attrs)
        for trait in traits.values():
            trait.instance_init(self)

    def set_trait(self, name: str, value: t.Any) -> None:
        """Forcibly sets trait attribute, including read-only attributes."""
        cls = self.__class__
        if not self.has_trait(name):
            raise TraitError(f'Class {cls.__name__} does not have a trait named {name}')
        getattr(cls, name).set(self, value)

    @classmethod
    def class_trait_names(cls: type[HasTraits], **metadata: t.Any) -> list[str]:
        """Get a list of all the names of this class' traits.

        This method is just like the :meth:`trait_names` method,
        but is unbound.
        """
        return list(cls.class_traits(**metadata))

    @classmethod
    def class_traits(cls: type[HasTraits], **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a ``dict`` of all the traits of this class.  The dictionary
        is keyed on the name and the values are the TraitType objects.

        This method is just like the :meth:`traits` method, but is unbound.

        The TraitTypes returned don't know anything about the values
        that the various HasTrait's instances are holding.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.
        """
        traits = cls._traits.copy()
        if len(metadata) == 0:
            return traits
        result = {}
        for name, trait in traits.items():
            for meta_name, meta_eval in metadata.items():
                if not callable(meta_eval):
                    meta_eval = _SimpleTest(meta_eval)
                if not meta_eval(trait.metadata.get(meta_name, None)):
                    break
            else:
                result[name] = trait
        return result

    @classmethod
    def class_own_traits(cls: type[HasTraits], **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a dict of all the traitlets defined on this class, not a parent.

        Works like `class_traits`, except for excluding traits from parents.
        """
        sup = super(cls, cls)
        return {n: t for n, t in cls.class_traits(**metadata).items() if getattr(sup, n, None) is not t}

    def has_trait(self, name: str) -> bool:
        """Returns True if the object has a trait with the specified name."""
        return name in self._traits

    def trait_has_value(self, name: str) -> bool:
        """Returns True if the specified trait has a value.

        This will return false even if ``getattr`` would return a
        dynamically generated default value. These default values
        will be recognized as existing only after they have been
        generated.

        Example

        .. code-block:: python

            class MyClass(HasTraits):
                i = Int()


            mc = MyClass()
            assert not mc.trait_has_value("i")
            mc.i  # generates a default value
            assert mc.trait_has_value("i")
        """
        return name in self._trait_values

    def trait_values(self, **metadata: t.Any) -> dict[str, t.Any]:
        """A ``dict`` of trait names and their values.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.

        Returns
        -------
        A ``dict`` of trait names and their values.

        Notes
        -----
        Trait values are retrieved via ``getattr``, any exceptions raised
        by traits or the operations they may trigger will result in the
        absence of a trait value in the result ``dict``.
        """
        return {name: getattr(self, name) for name in self.trait_names(**metadata)}

    def _get_trait_default_generator(self, name: str) -> t.Any:
        """Return default generator for a given trait

        Walk the MRO to resolve the correct default generator according to inheritance.
        """
        method_name = '_%s_default' % name
        if method_name in self.__dict__:
            return getattr(self, method_name)
        if method_name in self.__class__.__dict__:
            return getattr(self.__class__, method_name)
        return self._all_trait_default_generators[name]

    def trait_defaults(self, *names: str, **metadata: t.Any) -> dict[str, t.Any] | Sentinel:
        """Return a trait's default value or a dictionary of them

        Notes
        -----
        Dynamically generated default values may
        depend on the current state of the object."""
        for n in names:
            if not self.has_trait(n):
                raise TraitError(f"'{n}' is not a trait of '{type(self).__name__}' instances")
        if len(names) == 1 and len(metadata) == 0:
            return t.cast(Sentinel, self._get_trait_default_generator(names[0])(self))
        trait_names = self.trait_names(**metadata)
        trait_names.extend(names)
        defaults = {}
        for n in trait_names:
            defaults[n] = self._get_trait_default_generator(n)(self)
        return defaults

    def trait_names(self, **metadata: t.Any) -> list[str]:
        """Get a list of all the names of this class' traits."""
        return list(self.traits(**metadata))

    def traits(self, **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a ``dict`` of all the traits of this class.  The dictionary
        is keyed on the name and the values are the TraitType objects.

        The TraitTypes returned don't know anything about the values
        that the various HasTrait's instances are holding.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.
        """
        traits = self._traits.copy()
        if len(metadata) == 0:
            return traits
        result = {}
        for name, trait in traits.items():
            for meta_name, meta_eval in metadata.items():
                if not callable(meta_eval):
                    meta_eval = _SimpleTest(meta_eval)
                if not meta_eval(trait.metadata.get(meta_name, None)):
                    break
            else:
                result[name] = trait
        return result

    def trait_metadata(self, traitname: str, key: str, default: t.Any=None) -> t.Any:
        """Get metadata values for trait by key."""
        try:
            trait = getattr(self.__class__, traitname)
        except AttributeError as e:
            raise TraitError(f'Class {self.__class__.__name__} does not have a trait named {traitname}') from e
        metadata_name = '_' + traitname + '_metadata'
        if hasattr(self, metadata_name) and key in getattr(self, metadata_name):
            return getattr(self, metadata_name).get(key, default)
        else:
            return trait.metadata.get(key, default)

    @classmethod
    def class_own_trait_events(cls: type[HasTraits], name: str) -> dict[str, EventHandler]:
        """Get a dict of all event handlers defined on this class, not a parent.

        Works like ``event_handlers``, except for excluding traits from parents.
        """
        sup = super(cls, cls)
        return {n: e for n, e in cls.events(name).items() if getattr(sup, n, None) is not e}

    @classmethod
    def trait_events(cls: type[HasTraits], name: str | None=None) -> dict[str, EventHandler]:
        """Get a ``dict`` of all the event handlers of this class.

        Parameters
        ----------
        name : str (default: None)
            The name of a trait of this class. If name is ``None`` then all
            the event handlers of this class will be returned instead.

        Returns
        -------
        The event handlers associated with a trait name, or all event handlers.
        """
        events = {}
        for k, v in getmembers(cls):
            if isinstance(v, EventHandler):
                if name is None:
                    events[k] = v
                elif name in v.trait_names:
                    events[k] = v
                elif hasattr(v, 'tags'):
                    if cls.trait_names(**v.tags):
                        events[k] = v
        return events