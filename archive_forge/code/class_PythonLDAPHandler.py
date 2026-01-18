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
class PythonLDAPHandler(LDAPHandler):
    """LDAPHandler implementation which calls the python-ldap API.

    Note, the python-ldap API requires all string attribute values to be UTF-8
    encoded.

    Note, in python-ldap some fields (DNs, RDNs, attribute names, queries)
    are represented as text (str on Python 3, unicode on Python 2 when
    bytes_mode=False). For more details see:
    http://www.python-ldap.org/en/latest/bytes_mode.html#bytes-mode

    The KeystoneLDAPHandler enforces this prior to invoking the methods in this
    class.

    """

    def connect(self, url, page_size=0, alias_dereferencing=None, use_tls=False, tls_cacertfile=None, tls_cacertdir=None, tls_req_cert=ldap.OPT_X_TLS_DEMAND, chase_referrals=None, debug_level=None, conn_timeout=None, use_pool=None, pool_size=None, pool_retry_max=None, pool_retry_delay=None, pool_conn_timeout=None, pool_conn_lifetime=None):
        _common_ldap_initialization(url=url, use_tls=use_tls, tls_cacertfile=tls_cacertfile, tls_cacertdir=tls_cacertdir, tls_req_cert=tls_req_cert, debug_level=debug_level, timeout=conn_timeout)
        self.conn = ldap.initialize(url)
        self.conn.protocol_version = ldap.VERSION3
        if alias_dereferencing is not None:
            self.conn.set_option(ldap.OPT_DEREF, alias_dereferencing)
        self.page_size = page_size
        if use_tls:
            self.conn.start_tls_s()
        if chase_referrals is not None:
            self.conn.set_option(ldap.OPT_REFERRALS, int(chase_referrals))

    def set_option(self, option, invalue):
        return self.conn.set_option(option, invalue)

    def get_option(self, option):
        return self.conn.get_option(option)

    def simple_bind_s(self, who='', cred='', serverctrls=None, clientctrls=None):
        return self.conn.simple_bind_s(who, cred, serverctrls, clientctrls)

    def unbind_s(self):
        return self.conn.unbind_s()

    def add_s(self, dn, modlist):
        return self.conn.add_s(dn, modlist)

    def search_s(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0):
        return self.conn.search_s(base, scope, filterstr, attrlist, attrsonly)

    def search_ext(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0, serverctrls=None, clientctrls=None, timeout=-1, sizelimit=0):
        return self.conn.search_ext(base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)

    def result3(self, msgid=ldap.RES_ANY, all=1, timeout=None, resp_ctrl_classes=None):
        return self.conn.result3(msgid, all, timeout)

    def modify_s(self, dn, modlist):
        return self.conn.modify_s(dn, modlist)