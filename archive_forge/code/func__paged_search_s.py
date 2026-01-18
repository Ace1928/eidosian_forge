import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def _paged_search_s(self, base, scope, filterstr, attrlist=None):
    res = []
    use_old_paging_api = False
    if hasattr(ldap, 'LDAP_CONTROL_PAGE_OID'):
        use_old_paging_api = True
        lc = ldap.controls.SimplePagedResultsControl(controlType=ldap.LDAP_CONTROL_PAGE_OID, criticality=True, controlValue=(self.page_size, ''))
        page_ctrl_oid = ldap.LDAP_CONTROL_PAGE_OID
    else:
        lc = ldap.controls.libldap.SimplePagedResultsControl(criticality=True, size=self.page_size, cookie='')
        page_ctrl_oid = ldap.controls.SimplePagedResultsControl.controlType
    message = self.conn.search_ext(base, scope, filterstr, attrlist, serverctrls=[lc])
    while True:
        rtype, rdata, rmsgid, serverctrls = self.conn.result3(message)
        res.extend(rdata)
        pctrls = [c for c in serverctrls if c.controlType == page_ctrl_oid]
        if pctrls:
            if use_old_paging_api:
                est, cookie = pctrls[0].controlValue
                lc.controlValue = (self.page_size, cookie)
            else:
                cookie = lc.cookie = pctrls[0].cookie
            if cookie:
                message = self.conn.search_ext(base, scope, filterstr, attrlist, serverctrls=[lc])
            else:
                break
        else:
            LOG.warning('LDAP Server does not support paging. Disable paging in keystone.conf to avoid this message.')
            self._disable_paging()
            break
    return res