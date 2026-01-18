from typing import (
from .constants import (
from .exceptions import (
from .utils import (
class CommandSet(object):
    """
    Base class for defining sets of commands to load in cmd2.

    ``with_default_category`` can be used to apply a default category to all commands in the CommandSet.

    ``do_``, ``help_``, and ``complete_`` functions differ only in that self is the CommandSet instead of the cmd2 app
    """

    def __init__(self) -> None:
        self._cmd: Optional[cmd2.Cmd] = None
        self._settables: Dict[str, Settable] = {}
        self._settable_prefix = self.__class__.__name__

    def on_register(self, cmd: 'cmd2.Cmd') -> None:
        """
        Called by cmd2.Cmd as the first step to registering a CommandSet. The commands defined in this class have
        not been added to the CLI object at this point. Subclasses can override this to perform any initialization
        requiring access to the Cmd object (e.g. configure commands and their parsers based on CLI state data).

        :param cmd: The cmd2 main application
        """
        if self._cmd is None:
            self._cmd = cmd
        else:
            raise CommandSetRegistrationError('This CommandSet has already been registered')

    def on_registered(self) -> None:
        """
        Called by cmd2.Cmd after a CommandSet is registered and all its commands have been added to the CLI.
        Subclasses can override this to perform custom steps related to the newly added commands (e.g. setting
        them to a disabled state).
        """
        pass

    def on_unregister(self) -> None:
        """
        Called by ``cmd2.Cmd`` as the first step to unregistering a CommandSet. Subclasses can override this to
        perform any cleanup steps which require their commands being registered in the CLI.
        """
        pass

    def on_unregistered(self) -> None:
        """
        Called by ``cmd2.Cmd`` after a CommandSet has been unregistered and all its commands removed from the CLI.
        Subclasses can override this to perform remaining cleanup steps.
        """
        self._cmd = None

    @property
    def settable_prefix(self) -> str:
        return self._settable_prefix

    @property
    def settables(self) -> Mapping[str, Settable]:
        return self._settables

    def add_settable(self, settable: Settable) -> None:
        """
        Convenience method to add a settable parameter to the CommandSet

        :param settable: Settable object being added
        """
        if self._cmd:
            if not self._cmd.always_prefix_settables:
                if settable.name in self._cmd.settables.keys() and settable.name not in self._settables.keys():
                    raise KeyError(f'Duplicate settable: {settable.name}')
            else:
                prefixed_name = f'{self._settable_prefix}.{settable.name}'
                if prefixed_name in self._cmd.settables.keys() and settable.name not in self._settables.keys():
                    raise KeyError(f'Duplicate settable: {settable.name}')
        self._settables[settable.name] = settable

    def remove_settable(self, name: str) -> None:
        """
        Convenience method for removing a settable parameter from the CommandSet

        :param name: name of the settable being removed
        :raises: KeyError if the Settable matches this name
        """
        try:
            del self._settables[name]
        except KeyError:
            raise KeyError(name + ' is not a settable parameter')