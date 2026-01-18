from ncclient.transport.parser import DefaultXMLParser
import sys
def add_additional_netconf_params(self, kwargs):
    """Add additional NETCONF parameters

        Accept a keyword-argument dictionary to add additional NETCONF
        parameters that may be in addition to those specified by the
        default and device specific handlers.

        Currently, only additional client specified capabilities are
        supported and will be appended to default and device specific
        capabilities.

        Args:
            kwargs: A dictionary of specific NETCONF parameters to
                apply in addition to those derived by default and
                device specific handlers.
        """
    self.capabilities = kwargs.pop('capabilities', [])