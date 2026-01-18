import abc
import keystone.conf
from keystone import exception
def _get_list_limit(self):
    return CONF.resource.list_limit or CONF.list_limit