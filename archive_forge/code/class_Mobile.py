from .command import Command
class Mobile:
    ConnectionType = _ConnectionType
    ALL_NETWORK = ConnectionType(6)
    WIFI_NETWORK = ConnectionType(2)
    DATA_NETWORK = ConnectionType(4)
    AIRPLANE_MODE = ConnectionType(1)

    def __init__(self, driver):
        import weakref
        self._driver = weakref.proxy(driver)

    @property
    def network_connection(self):
        return self.ConnectionType(self._driver.execute(Command.GET_NETWORK_CONNECTION)['value'])

    def set_network_connection(self, network):
        """Set the network connection for the remote device.

        Example of setting airplane mode::

            driver.mobile.set_network_connection(driver.mobile.AIRPLANE_MODE)
        """
        mode = network.mask if isinstance(network, self.ConnectionType) else network
        return self.ConnectionType(self._driver.execute(Command.SET_NETWORK_CONNECTION, {'name': 'network_connection', 'parameters': {'type': mode}})['value'])

    @property
    def context(self):
        """Returns the current context (Native or WebView)."""
        return self._driver.execute(Command.CURRENT_CONTEXT_HANDLE)

    @context.setter
    def context(self, new_context) -> None:
        """Sets the current context."""
        self._driver.execute(Command.SWITCH_TO_CONTEXT, {'name': new_context})

    @property
    def contexts(self):
        """Returns a list of available contexts."""
        return self._driver.execute(Command.CONTEXT_HANDLES)