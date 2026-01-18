from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAPQuery(object):

    def __init__(self, qry):
        self.qry = qry

    def build_request(self, attributes):
        params = copy.deepcopy(self.qry)
        params['attrlist'] = attributes
        return params

    def ldap_search(self, connection, required_attributes):
        query = self.build_request(required_attributes)
        derefAlias = query.pop('derefAlias', None)
        if derefAlias:
            ldap.set_option(ldap.OPT_DEREF, derefAlias)
        try:
            result = connection.search_ext_s(**query)
            if not result or len(result) == 0:
                return (None, "Entry not found for base='{0}' and filter='{1}'".format(query['base'], query['filterstr']))
            return (result, None)
        except ldap.NO_SUCH_OBJECT:
            return (None, "search for entry with base dn='{0}' refers to a non-existent entry".format(query['base']))