import urllib.parse
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_parents_as_ids_and_parents_as_list(self, parents_as_ids, parents_as_list):
    if parents_as_ids and parents_as_list:
        msg = _('Specify either parents_as_ids or parents_as_list parameters, not both')
        raise exceptions.ValidationError(msg)