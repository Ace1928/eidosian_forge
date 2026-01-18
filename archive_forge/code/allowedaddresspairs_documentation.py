from oslo_config import cfg
from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib.exceptions import allowedaddresspairs as exceptions
Validates a list of allowed address pair dicts.

    Validation herein requires the caller to have registered the
    max_allowed_address_pair oslo config option in the global CONF prior
    to having this validator used.

    :param address_pairs: A list of address pair dicts to validate.
    :param valid_values: Not used.
    :returns: None
    :raises: AllowedAddressPairExhausted if the address pairs requested
    exceeds cfg.CONF.max_allowed_address_pair. AllowedAddressPairsMissingIP
    if any address pair dicts are missing and IP address.
    DuplicateAddressPairInRequest if duplicated IPs are in the list of
    address pair dicts. Otherwise a HTTPBadRequest is raised if any of
    the address pairs are invalid.
    