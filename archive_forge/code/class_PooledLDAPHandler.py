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
class PooledLDAPHandler(LDAPHandler):
    """LDAPHandler implementation which uses pooled connection manager.

    Pool specific configuration is defined in [ldap] section.
    All other LDAP configuration is still used from [ldap] section

    Keystone LDAP authentication logic authenticates an end user using its DN
    and password via LDAP bind to establish supplied password is correct.
    This can fill up the pool quickly (as pool re-uses existing connection
    based on its bind data) and would not leave space in pool for connection
    re-use for other LDAP operations.
    Now a separate pool can be established for those requests when related flag
    'use_auth_pool' is enabled. That pool can have its own size and
    connection lifetime. Other pool attributes are shared between those pools.
    If 'use_pool' is disabled, then 'use_auth_pool' does not matter.
    If 'use_auth_pool' is not enabled, then connection pooling is not used for
    those LDAP operations.

    Note, the python-ldap API requires all string attribute values to be UTF-8
    encoded. The KeystoneLDAPHandler enforces this prior to invoking the
    methods in this class.

    Note, in python-ldap some fields (DNs, RDNs, attribute names, queries)
    are represented as text (str on Python 3, unicode on Python 2 when
    bytes_mode=False). For more details see:
    http://www.python-ldap.org/en/latest/bytes_mode.html#bytes-mode

    """
    Connector = ldappool.StateConnector
    auth_pool_prefix = 'auth_pool_'
    connection_pools = {}

    def __init__(self, conn=None, use_auth_pool=False):
        super(PooledLDAPHandler, self).__init__(conn=conn)
        self.who = ''
        self.cred = ''
        self.conn_options = {}
        self.page_size = None
        self.use_auth_pool = use_auth_pool
        self.conn_pool = None

    def connect(self, url, page_size=0, alias_dereferencing=None, use_tls=False, tls_cacertfile=None, tls_cacertdir=None, tls_req_cert=ldap.OPT_X_TLS_DEMAND, chase_referrals=None, debug_level=None, conn_timeout=None, use_pool=None, pool_size=None, pool_retry_max=None, pool_retry_delay=None, pool_conn_timeout=None, pool_conn_lifetime=None):
        _common_ldap_initialization(url=url, use_tls=use_tls, tls_cacertfile=tls_cacertfile, tls_cacertdir=tls_cacertdir, tls_req_cert=tls_req_cert, debug_level=debug_level, timeout=pool_conn_timeout)
        self.page_size = page_size
        if alias_dereferencing is not None:
            self.set_option(ldap.OPT_DEREF, alias_dereferencing)
        if chase_referrals is not None:
            self.set_option(ldap.OPT_REFERRALS, int(chase_referrals))
        if self.use_auth_pool:
            pool_url = self.auth_pool_prefix + url
        else:
            pool_url = url
        try:
            self.conn_pool = self.connection_pools[pool_url]
        except KeyError:
            self.conn_pool = ldappool.ConnectionManager(url, size=pool_size, retry_max=pool_retry_max, retry_delay=pool_retry_delay, timeout=pool_conn_timeout, connector_cls=self.Connector, use_tls=use_tls, max_lifetime=pool_conn_lifetime)
            self.connection_pools[pool_url] = self.conn_pool

    def set_option(self, option, invalue):
        self.conn_options[option] = invalue

    def get_option(self, option):
        value = self.conn_options.get(option)
        if value is None:
            with self._get_pool_connection() as conn:
                value = conn.get_option(option)
        return value

    def _apply_options(self, conn):
        if conn.get_lifetime() > 30:
            return
        for option, invalue in self.conn_options.items():
            conn.set_option(option, invalue)

    def _get_pool_connection(self):
        return self.conn_pool.connection(self.who, self.cred)

    def simple_bind_s(self, who='', cred='', serverctrls=None, clientctrls=None):
        self.who = who
        self.cred = cred
        with self._get_pool_connection() as conn:
            self._apply_options(conn)

    def unbind_s(self):
        pass

    @use_conn_pool
    def add_s(self, conn, dn, modlist):
        return conn.add_s(dn, modlist)

    @use_conn_pool
    def search_s(self, conn, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0):
        return conn.search_s(base, scope, filterstr, attrlist, attrsonly)

    def search_ext(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0, serverctrls=None, clientctrls=None, timeout=-1, sizelimit=0):
        """Return an AsynchronousMessage instance, it asynchronous API.

        The AsynchronousMessage instance can be safely used in a call to
        `result3()`.

        To work with `result3()` API in predictable manner, the same LDAP
        connection is needed which originally provided the `msgid`. So, this
        method wraps the existing connection and `msgid` in a new
        `AsynchronousMessage` instance. The connection associated with
        `search_ext()` is released after `result3()` fetches the data
        associated with `msgid`.

        """
        conn_ctxt = self._get_pool_connection()
        conn = conn_ctxt.__enter__()
        try:
            msgid = conn.search_ext(base, scope, filterstr, attrlist, attrsonly, serverctrls, clientctrls, timeout, sizelimit)
        except Exception:
            conn_ctxt.__exit__(*sys.exc_info())
            raise
        return AsynchronousMessage(msgid, conn, conn_ctxt)

    def result3(self, message, all=1, timeout=None, resp_ctrl_classes=None):
        """Wait for and return the result to an asynchronous message.

        This method returns the result of an operation previously initiated by
        one of the LDAP asynchronous operation routines (e.g., `search_ext()`).
        The `search_ext()` method in python-ldap returns an invocation
        identifier, or a message ID, upon successful initiation of the
        operation by the LDAP server.

        The `message` is expected to be instance of class
        `AsynchronousMessage`, which contains the message ID and the connection
        used to make the original request.

        The connection and context manager associated with `search_ext()` are
        cleaned up when message.clean() is called.

        """
        try:
            results = message.connection.result3(message.id, all, timeout)
        finally:
            message.clean()
        return results

    @use_conn_pool
    def modify_s(self, conn, dn, modlist):
        return conn.modify_s(dn, modlist)