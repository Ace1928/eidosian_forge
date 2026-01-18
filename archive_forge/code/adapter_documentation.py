from keystoneauth1 import adapter
from keystoneauth1.loading import _utils
from keystoneauth1.loading import base
Create an Adapter object from an oslo_config object.

        The options must have been previously registered with
        register_conf_options.

        :param oslo_config.Cfg conf: config object to register with.
        :param string group: The ini group to register options in.
        :param dict kwargs: Additional parameters to pass to Adapter
                            construction.
        :returns: A new Adapter object.
        :rtype: :py:class:`.Adapter`
        