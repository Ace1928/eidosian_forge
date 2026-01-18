import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
Get config settings for a named client.

        Settings will also be looked for in a section called 'client'.
        If settings are found in both, they will be merged with the settings
        from the named section winning over the settings from client section,
        and both winning over provided defaults.

        :param string name:
            Name of the config section to look for.
        :param dict defaults:
            Default settings to use.

        :returns:
            A dict containing merged settings from the named section, the
            client section and the defaults.
        