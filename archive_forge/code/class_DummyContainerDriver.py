from libcloud.container.base import ContainerDriver
class DummyContainerDriver(ContainerDriver):
    """
    Dummy Container driver.

    >>> from libcloud.container.drivers.dummy import DummyContainerDriver
    >>> driver = DummyContainerDriver('key', 'secret')
    >>> driver.name
    'Dummy Container Provider'
    """
    name = 'Dummy Container Provider'
    website = 'http://example.com'
    supports_clusters = False

    def __init__(self, api_key, api_secret):
        """
        :param    api_key:    API key or username to used (required)
        :type     api_key:    ``str``

        :param    api_secret: Secret password to be used (required)
        :type     api_secret: ``str``

        :rtype: ``None``
        """