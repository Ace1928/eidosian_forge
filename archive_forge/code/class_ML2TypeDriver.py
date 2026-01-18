import abc
from neutron_lib.api.definitions import portbindings
class ML2TypeDriver(_TypeDriverBase, metaclass=abc.ABCMeta):
    """Define abstract interface for ML2 type drivers.

    ML2 type drivers each support a specific network_type for provider
    and/or tenant network segments. Type drivers must implement this
    abstract interface, which defines the API by which the plugin uses
    the driver to manage the persistent type-specific resource
    allocation state associated with network segments of that type.

    Network segments are represented by segment dictionaries using the
    NETWORK_TYPE, PHYSICAL_NETWORK, and SEGMENTATION_ID keys defined
    above, corresponding to the provider attributes.  Future revisions
    of the TypeDriver API may add additional segment dictionary
    keys. Attributes not applicable for a particular network_type may
    either be excluded or stored as None.

    ML2TypeDriver passes context as argument for:
    - reserve_provider_segment
    - allocate_tenant_segment
    - release_segment
    - get_allocation
    """

    @abc.abstractmethod
    def reserve_provider_segment(self, context, segment, filters=None):
        """Reserve resource associated with a provider network segment.

        :param context: instance of neutron context with DB session
        :param segment: segment dictionary
        :param filters: a dictionary that is used as search criteria
        :returns: segment dictionary

        Called inside transaction context on session to reserve the
        type-specific resource for a provider network segment. The
        segment dictionary passed in was returned by a previous
        validate_provider_segment() call.
        """

    @abc.abstractmethod
    def allocate_tenant_segment(self, context, filters=None):
        """Allocate resource for a new tenant network segment.

        :param context: instance of neutron context with DB session
        :param filters: a dictionary that is used as search criteria
        :returns: segment dictionary using keys defined above

        Called inside transaction context on session to allocate a new
        tenant network, typically from a type-specific resource
        pool. If successful, return a segment dictionary describing
        the segment. If tenant network segment cannot be allocated
        (i.e. tenant networks not supported or resource pool is
        exhausted), return None.
        """

    @abc.abstractmethod
    def release_segment(self, context, segment):
        """Release network segment.

        :param context: instance of neutron context with DB session
        :param segment: segment dictionary using keys defined above

        Called inside transaction context on session to release a
        tenant or provider network's type-specific resource. Runtime
        errors are not expected, but raising an exception will result
        in rollback of the transaction.
        """

    @abc.abstractmethod
    def initialize_network_segment_range_support(self):
        """Perform driver network segment range initialization.

        Called during the initialization of the ``network-segment-range``
        service plugin if enabled, after all drivers have been loaded and the
        database has been initialized. This reloads the `default`
        network segment ranges when Neutron server starts/restarts.
        """

    @abc.abstractmethod
    def update_network_segment_range_allocations(self):
        """Update driver network segment range allocations.

        This syncs the driver segment allocations when network segment ranges
        have been created, updated or deleted.
        """