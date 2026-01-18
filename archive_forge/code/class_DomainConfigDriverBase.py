import abc
from keystone import exception
class DomainConfigDriverBase(object, metaclass=abc.ABCMeta):
    """Interface description for a Domain Config driver."""

    @abc.abstractmethod
    def create_config_options(self, domain_id, option_list):
        """Create config options for a domain.

        Any existing config options will first be deleted.

        :param domain_id: the domain for this option
        :param option_list: a list of dicts, each one specifying an option

        Option schema::

            type: dict
            properties:
                group:
                    type: string
                option:
                    type: string
                value:
                    type: depends on the option
                sensitive:
                    type: boolean
            required: [group, option, value, sensitive]
            additionalProperties: false

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_config_option(self, domain_id, group, option, sensitive=False):
        """Get the config option for a domain.

        :param domain_id: the domain for this option
        :param group: the group name
        :param option: the option name
        :param sensitive: whether the option is sensitive

        :returns: dict containing group, option and value
        :raises keystone.exception.DomainConfigNotFound: the option doesn't
                                                         exist.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_config_options(self, domain_id, group=None, option=False, sensitive=False):
        """Get a config options for a domain.

        :param domain_id: the domain for this option
        :param group: optional group option name
        :param option: optional option name. If group is None, then this
                       parameter is ignored
        :param sensitive: whether the option is sensitive

        :returns: list of dicts containing group, option and value

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def update_config_options(self, domain_id, option_list):
        """Update config options for a domain.

        :param domain_id: the domain for this option
        :param option_list: a list of dicts, each one specifying an option

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_config_options(self, domain_id, group=None, option=None):
        """Delete config options for a domain.

        Allows deletion of all options for a domain, all options in a group
        or a specific option. The driver is silent if there are no options
        to delete.

        :param domain_id: the domain for this option
        :param group: optional group option name
        :param option: optional option name. If group is None, then this
                       parameter is ignored

        The option is uniquely defined by domain_id, group and option,
        irrespective of whether it is sensitive ot not.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def obtain_registration(self, domain_id, type):
        """Try and register this domain to use the type specified.

        :param domain_id: the domain required
        :param type: type of registration
        :returns: True if the domain was registered, False otherwise. Failing
                  to register means that someone already has it (which could
                  even be the domain being requested).

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def read_registration(self, type):
        """Get the domain ID of who is registered to use this type.

        :param type: type of registration
        :returns: domain_id of who is registered.
        :raises keystone.exception.ConfigRegistrationNotFound: If nobody is
            registered.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def release_registration(self, domain_id, type=None):
        """Release registration if it is held by the domain specified.

        If the specified domain is registered for this domain then free it,
        if it is not then do nothing - no exception is raised.

        :param domain_id: the domain in question
        :param type: type of registration, if None then all registrations
                     for this domain will be freed

        """
        raise exception.NotImplemented()