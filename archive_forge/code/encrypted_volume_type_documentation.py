from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource for encrypting a cinder volume type.

    A Volume Encryption Type is a collection of settings used to conduct
    encryption for a specific volume type.

    Note that default cinder security policy usage of this resource
    is limited to being used by administrators only.
    