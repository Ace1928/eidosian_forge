from keystoneauth1.exceptions import base
class NoMatchingPlugin(AuthPluginException):
    """No auth plugins could be created from the parameters provided.

    :param str name: The name of the plugin that was attempted to load.

    .. py:attribute:: name

        The name of the plugin that was attempted to load.
    """

    def __init__(self, name):
        self.name = name
        msg = 'The plugin %s could not be found' % name
        super(NoMatchingPlugin, self).__init__(msg)