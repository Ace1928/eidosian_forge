import abc
class ConfigurationSource(metaclass=abc.ABCMeta):
    """A configuration source option for oslo.config.

    A configuration source is able to fetch configuration values based on
    a (group, option) key from an external source that supports key-value
    mapping such as INI files, key-value stores, secret stores, and so on.

    """

    @abc.abstractmethod
    def get(self, group_name, option_name, opt):
        """Return the value of the option from the group.

        :param group_name: Name of the group.
        :type group_name: str
        :param option_name: Name of the option.
        :type option_name: str
        :param opt: The option definition.
        :type opt: Opt
        :returns: A tuple (value, location) where value is the option value
                  or oslo_config.sources._NoValue if the (group, option) is
                  not present in the source, and location is a LocationInfo.
        """