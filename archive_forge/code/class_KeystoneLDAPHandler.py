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
class KeystoneLDAPHandler(LDAPHandler):
    """Convert data types and perform logging.

    This LDAP interface wraps the python-ldap based interfaces. The
    python-ldap interfaces require string values encoded in UTF-8 with
    the exception of [1]. The OpenStack logging framework at the time
    of this writing is not capable of accepting strings encoded in
    UTF-8, the log functions will throw decoding errors if a non-ascii
    character appears in a string.

    [1] In python-ldap, some fields (DNs, RDNs, attribute names,
    queries) are represented as text (str on Python 3, unicode on
    Python 2 when bytes_mode=False). For more details see:
    http://www.python-ldap.org/en/latest/bytes_mode.html#bytes-mode

    Prior to the call Python data types are converted to a string
    representation as required by the LDAP APIs.

    Then logging is performed so we can track what is being
    sent/received from LDAP. Also the logging filters security
    sensitive items (i.e. passwords).

    Then the string values are encoded into UTF-8.

    Then the LDAP API entry point is invoked.

    Data returned from the LDAP call is converted back from UTF-8
    encoded strings into the Python data type used internally in
    OpenStack.

    """

    def __init__(self, conn=None):
        super(KeystoneLDAPHandler, self).__init__(conn=conn)
        self.page_size = 0

    def __enter__(self):
        """Enter runtime context."""
        return self

    def _disable_paging(self):
        self.page_size = 0

    def connect(self, url, page_size=0, alias_dereferencing=None, use_tls=False, tls_cacertfile=None, tls_cacertdir=None, tls_req_cert=ldap.OPT_X_TLS_DEMAND, chase_referrals=None, debug_level=None, conn_timeout=None, use_pool=None, pool_size=None, pool_retry_max=None, pool_retry_delay=None, pool_conn_timeout=None, pool_conn_lifetime=None):
        self.page_size = page_size
        return self.conn.connect(url, page_size, alias_dereferencing, use_tls, tls_cacertfile, tls_cacertdir, tls_req_cert, chase_referrals, debug_level=debug_level, conn_timeout=conn_timeout, use_pool=use_pool, pool_size=pool_size, pool_retry_max=pool_retry_max, pool_retry_delay=pool_retry_delay, pool_conn_timeout=pool_conn_timeout, pool_conn_lifetime=pool_conn_lifetime)

    def set_option(self, option, invalue):
        return self.conn.set_option(option, invalue)

    def get_option(self, option):
        return self.conn.get_option(option)

    def simple_bind_s(self, who='', cred='', serverctrls=None, clientctrls=None):
        LOG.debug('LDAP bind: who=%s', who)
        return self.conn.simple_bind_s(who, cred, serverctrls=serverctrls, clientctrls=clientctrls)

    def unbind_s(self):
        LOG.debug('LDAP unbind')
        return self.conn.unbind_s()

    def add_s(self, dn, modlist):
        ldap_attrs = [(kind, [py2ldap(x) for x in safe_iter(values)]) for kind, values in modlist]
        logging_attrs = [(kind, values if kind != 'userPassword' else ['****']) for kind, values in ldap_attrs]
        LOG.debug('LDAP add: dn=%s attrs=%s', dn, logging_attrs)
        ldap_attrs_utf8 = [(kind, [utf8_encode(x) for x in safe_iter(values)]) for kind, values in ldap_attrs]
        return self.conn.add_s(dn, ldap_attrs_utf8)

    def search_s(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0):
        if attrlist is not None:
            attrlist = [attr for attr in attrlist if attr is not None]
        LOG.debug('LDAP search: base=%s scope=%s filterstr=%s attrs=%s attrsonly=%s', base, scope, filterstr, attrlist, attrsonly)
        if self.page_size:
            ldap_result = self._paged_search_s(base, scope, filterstr, attrlist)
        else:
            try:
                ldap_result = self.conn.search_s(base, scope, filterstr, attrlist, attrsonly)
            except ldap.SIZELIMIT_EXCEEDED:
                raise exception.LDAPSizeLimitExceeded()
        py_result = convert_ldap_result(ldap_result)
        return py_result

    def search_ext(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0, serverctrls=None, clientctrls=None, timeout=-1, sizelimit=0):
        if attrlist is not None:
            attrlist = [attr for attr in attrlist if attr is not None]
        LOG.debug('LDAP search_ext: base=%s scope=%s filterstr=%s attrs=%s attrsonly=%s serverctrls=%s clientctrls=%s timeout=%s sizelimit=%s', base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)
        return self.conn.search_ext(base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)

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

    def result3(self, msgid=ldap.RES_ANY, all=1, timeout=None, resp_ctrl_classes=None):
        ldap_result = self.conn.result3(msgid, all, timeout, resp_ctrl_classes)
        LOG.debug('LDAP result3: msgid=%s all=%s timeout=%s resp_ctrl_classes=%s ldap_result=%s', msgid, all, timeout, resp_ctrl_classes, ldap_result)
        rtype, rdata, rmsgid, serverctrls = ldap_result
        py_result = convert_ldap_result(rdata)
        return py_result

    def modify_s(self, dn, modlist):
        ldap_modlist = [(op, kind, None if values is None else [py2ldap(x) for x in safe_iter(values)]) for op, kind, values in modlist]
        logging_modlist = [(op, kind, values if kind != 'userPassword' else ['****']) for op, kind, values in ldap_modlist]
        LOG.debug('LDAP modify: dn=%s modlist=%s', dn, logging_modlist)
        ldap_modlist_utf8 = [(op, kind, None if values is None else [utf8_encode(x) for x in safe_iter(values)]) for op, kind, values in ldap_modlist]
        return self.conn.modify_s(dn, ldap_modlist_utf8)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit runtime context, unbind LDAP."""
        self.unbind_s()