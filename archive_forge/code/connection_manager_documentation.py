import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
Get swift endpoint from keystone

        Return endpoint for swift from service catalog if not overridden in
        store configuration. The method works only Keystone v3.
        If you are using different version (1 or 2)
        it returns None.
        :return: swift endpoint
        