import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def _paren_groups(source):
    """Split a string into parenthesized groups."""
    count = 0
    start = 0
    result = []
    for pos in range(len(source)):
        if source[pos] == '(':
            if count == 0:
                start = pos
            count += 1
        if source[pos] == ')':
            count -= 1
            if count == 0:
                result.append(source[start:pos + 1])
    return result