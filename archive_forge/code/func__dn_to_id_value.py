import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def _dn_to_id_value(self, dn):
    return ldap.dn.str2dn(dn)[0][0][1]