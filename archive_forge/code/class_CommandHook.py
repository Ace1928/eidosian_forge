import abc
class CommandHook(object, metaclass=abc.ABCMeta):
    """Base class for command hooks.

    :param app: Command instance being invoked
    :paramtype app: cliff.command.Command

    """

    def __init__(self, command):
        self.cmd = command

    @abc.abstractmethod
    def get_parser(self, parser):
        """Return an :class:`argparse.ArgumentParser`.

        :param parser: An existing ArgumentParser instance to be modified.
        :paramtype parser: ArgumentParser
        :returns: ArgumentParser
        """
        return parser

    @abc.abstractmethod
    def get_epilog(self):
        """Return text to add to the command help epilog."""
        return ''

    @abc.abstractmethod
    def before(self, parsed_args):
        """Called before the command's take_action() method.

        :param parsed_args: The arguments to the command.
        :paramtype parsed_args: argparse.Namespace
        :returns: argparse.Namespace
        """
        return parsed_args

    @abc.abstractmethod
    def after(self, parsed_args, return_code):
        """Called after the command's take_action() method.

        :param parsed_args: The arguments to the command.
        :paramtype parsed_args: argparse.Namespace
        :param return_code: The value returned from take_action().
        :paramtype return_code: int
        :returns: int
        """
        return return_code