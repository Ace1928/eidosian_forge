from abc import ABCMeta, abstractmethod
class SubscriberResetHandler(metaclass=ABCMeta):
    """Helps to reset subscriber state when the `RESET` signal is received from the server."""

    @abstractmethod
    async def handle_reset(self):
        """Reset subscriber state.

        Raises:
            GoogleAPICallError: If reset handling fails. The subscriber will shut down.
        """
        raise NotImplementedError()