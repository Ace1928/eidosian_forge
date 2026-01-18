import abc
from neutron_lib.api.definitions import portbindings
class ExtensionDriver(object, metaclass=abc.ABCMeta):
    """Define stable abstract interface for ML2 extension drivers.

    An extension driver extends the core resources implemented by the
    ML2 plugin with additional attributes. Methods that process create
    and update operations for these resources validate and persist
    values for extended attributes supplied through the API. Other
    methods extend the resource dictionaries returned from the API
    operations with the values of the extended attributes.
    """

    @abc.abstractmethod
    def initialize(self):
        """Perform driver initialization.

        Called after all drivers have been loaded and the database has
        been initialized. No abstract methods defined below will be
        called prior to this method being called.
        """

    @property
    def extension_alias(self):
        """Supported extension alias.

        Return the alias identifying the core API extension supported
        by this driver. Do not declare if API extension handling will
        be left to a service plugin, and we just need to provide
        core resource extension and updates.
        """
        return

    @property
    def extension_aliases(self):
        """List of extension aliases supported by the driver.

        Return a list of aliases identifying the core API extensions
        supported by the driver. By default this just returns the
        extension_alias property for backwards compatibility.
        """
        return [self.extension_alias]

    def process_create_network(self, plugin_context, data, result):
        """Process extended attributes for create network.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming network data
        :param result: network dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and persist any extended network attributes defined by this
        driver. Extended attribute values must also be added to
        result.
        """
        pass

    def process_create_subnet(self, plugin_context, data, result):
        """Process extended attributes for create subnet.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming subnet data
        :param result: subnet dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and persist any extended subnet attributes defined by this
        driver. Extended attribute values must also be added to
        result.
        """
        pass

    def process_create_port(self, plugin_context, data, result):
        """Process extended attributes for create port.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming port data
        :param result: port dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and persist any extended port attributes defined by this
        driver. Extended attribute values must also be added to
        result.
        """
        pass

    def process_update_network(self, plugin_context, data, result):
        """Process extended attributes for update network.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming network data
        :param result: network dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and update any extended network attributes defined by this
        driver. Extended attribute values, whether updated or not,
        must also be added to result.
        """
        pass

    def process_update_subnet(self, plugin_context, data, result):
        """Process extended attributes for update subnet.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming subnet data
        :param result: subnet dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and update any extended subnet attributes defined by this
        driver. Extended attribute values, whether updated or not,
        must also be added to result.
        """
        pass

    def process_update_port(self, plugin_context, data, result):
        """Process extended attributes for update port.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming port data
        :param result: port dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and update any extended port attributes defined by this
        driver. Extended attribute values, whether updated or not,
        must also be added to result.
        """
        pass

    def extend_network_dict(self, session, base_model, result):
        """Add extended attributes to network dictionary.

        :param session: database session
        :param base_model: network model data
        :param result: network dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a network
        dictionary to be used for mechanism driver calls and/or
        returned as the result of a network operation.
        """
        pass

    def extend_subnet_dict(self, session, base_model, result):
        """Add extended attributes to subnet dictionary.

        :param session: database session
        :param base_model: subnet model data
        :param result: subnet dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a subnet
        dictionary to be used for mechanism driver calls and/or
        returned as the result of a subnet operation.
        """
        pass

    def extend_port_dict(self, session, base_model, result):
        """Add extended attributes to port dictionary.

        :param session: database session
        :param base_model: port model data
        :param result: port dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a port
        dictionary to be used for mechanism driver calls
        and/or returned as the result of a port operation.
        """
        pass