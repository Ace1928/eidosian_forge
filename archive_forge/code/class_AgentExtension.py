import abc
class AgentExtension(object, metaclass=abc.ABCMeta):
    """Define stable abstract interface for agent extensions.

    An agent extension extends the agent core functionality.
    """

    @abc.abstractmethod
    def initialize(self, connection, driver_type):
        """Perform agent core resource extension initialization.

        :param connection: RPC connection that can be reused by the extension
                           to define its RPC endpoints
        :param driver_type: String that defines the agent type to the
                            extension. Can be used to choose the right backend
                            implementation.

        Called after all extensions have been loaded.
        No resource (port, policy, router, etc.) handling will be called before
        this method.
        """

    def consume_api(self, agent_api):
        """Consume the AgentAPI instance from the AgentExtensionsManager.

        Allows an extension to gain access to resources internal to the
        neutron agent and otherwise unavailable to the extension.  Examples of
        such resources include bridges, ports, and routers.

        :param agent_api: An instance of an agent-specific API.
        """