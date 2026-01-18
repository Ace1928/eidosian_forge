import abc
class HBChannelABC(ChannelABC):
    """HBChannel ABC.

    The docstrings for this class can be found in the base implementation:

    `jupyter_client.channels.HBChannel`
    """

    @abc.abstractproperty
    def time_to_dead(self) -> float:
        pass

    @abc.abstractmethod
    def pause(self) -> None:
        """Pause the heartbeat channel."""
        pass

    @abc.abstractmethod
    def unpause(self) -> None:
        """Unpause the heartbeat channel."""
        pass

    @abc.abstractmethod
    def is_beating(self) -> bool:
        """Test whether the channel is beating."""
        pass