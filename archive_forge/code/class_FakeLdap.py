import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
class FakeLdap(common.LDAPHandler):
    """Emulate the python-ldap API.

    The python-ldap API requires all strings to be UTF-8 encoded with the
    exception of [1]. This is assured by the caller of this interface
    (i.e. KeystoneLDAPHandler).

    However, internally this emulation MUST process and store strings
    in a canonical form which permits operations on
    characters. Encoded strings do not provide the ability to operate
    on characters. Therefore this emulation accepts UTF-8 encoded
    strings, decodes them to unicode for operations internal to this
    emulation, and encodes them back to UTF-8 when returning values
    from the emulation.

    [1] Some fields (DNs, RDNs, attribute names, queries) are represented
    as text in python-ldap for Python 3, and for Python 2 when
    bytes_mode=False. For more details see:
    http://www.python-ldap.org/en/latest/bytes_mode.html#bytes-mode

    """
    __prefix = 'ldap:'

    def __init__(self, conn=None):
        super(FakeLdap, self).__init__(conn=conn)
        self._ldap_options = {ldap.OPT_DEREF: ldap.DEREF_NEVER}

    def connect(self, url, page_size=0, alias_dereferencing=None, use_tls=False, tls_cacertfile=None, tls_cacertdir=None, tls_req_cert='demand', chase_referrals=None, debug_level=None, use_pool=None, pool_size=None, pool_retry_max=None, pool_retry_delay=None, pool_conn_timeout=None, pool_conn_lifetime=None, conn_timeout=None):
        if url.startswith('fake://memory'):
            if url not in FakeShelves:
                FakeShelves[url] = FakeShelve()
            self.db = FakeShelves[url]
        else:
            self.db = shelve.open(url[7:])
        using_ldaps = url.lower().startswith('ldaps')
        if use_tls and using_ldaps:
            raise AssertionError('Invalid TLS / LDAPS combination')
        if use_tls:
            if tls_cacertfile:
                ldap.set_option(ldap.OPT_X_TLS_CACERTFILE, tls_cacertfile)
            elif tls_cacertdir:
                ldap.set_option(ldap.OPT_X_TLS_CACERTDIR, tls_cacertdir)
            if tls_req_cert in list(common.LDAP_TLS_CERTS.values()):
                ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, tls_req_cert)
            else:
                raise ValueError('invalid TLS_REQUIRE_CERT tls_req_cert=%s', tls_req_cert)
        if alias_dereferencing is not None:
            self.set_option(ldap.OPT_DEREF, alias_dereferencing)
        self.page_size = page_size
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.pool_retry_max = pool_retry_max
        self.pool_retry_delay = pool_retry_delay
        self.pool_conn_timeout = pool_conn_timeout
        self.pool_conn_lifetime = pool_conn_lifetime
        self.conn_timeout = conn_timeout

    def _dn_to_id_attr(self, dn):
        return ldap.dn.str2dn(dn)[0][0][0]

    def _dn_to_id_value(self, dn):
        return ldap.dn.str2dn(dn)[0][0][1]

    def key(self, dn):
        return '%s%s' % (self.__prefix, dn)

    def simple_bind_s(self, who='', cred='', serverctrls=None, clientctrls=None):
        """Provide for compatibility but this method is ignored."""
        if server_fail:
            raise ldap.SERVER_DOWN
        whos = ['cn=Admin', CONF.ldap.user]
        if who in whos and cred in ['password', CONF.ldap.password]:
            self.connected = True
            self.who = who
            self.cred = cred
            return
        attrs = self.db.get(self.key(who))
        if not attrs:
            LOG.debug('who=%s not found, binding anonymously', who)
        db_password = ''
        if attrs:
            try:
                db_password = attrs['userPassword'][0]
            except (KeyError, IndexError):
                LOG.debug('bind fail: password for who=%s not found', who)
                raise ldap.INAPPROPRIATE_AUTH
        if cred != db_password:
            LOG.debug('bind fail: password for who=%s does not match', who)
            raise ldap.INVALID_CREDENTIALS

    def unbind_s(self):
        """Provide for compatibility but this method is ignored."""
        self.connected = False
        self.who = None
        self.cred = None
        if server_fail:
            raise ldap.SERVER_DOWN

    def add_s(self, dn, modlist):
        """Add an object with the specified attributes at dn."""
        if server_fail:
            raise ldap.SERVER_DOWN
        id_attr_in_modlist = False
        id_attr = self._dn_to_id_attr(dn)
        id_value = self._dn_to_id_value(dn)
        for k, dummy_v in modlist:
            if k is None:
                raise TypeError('must be string, not None. modlist=%s' % modlist)
            if k == id_attr:
                for val in dummy_v:
                    if common.utf8_decode(val) == id_value:
                        id_attr_in_modlist = True
        if not id_attr_in_modlist:
            LOG.debug('id_attribute=%(attr)s missing, attributes=%(attrs)s', {'attr': id_attr, 'attrs': modlist})
            raise ldap.NAMING_VIOLATION
        key = self.key(dn)
        LOG.debug('add item: dn=%(dn)s, attrs=%(attrs)s', {'dn': dn, 'attrs': modlist})
        if key in self.db:
            LOG.debug('add item failed: dn=%s is already in store.', dn)
            raise ldap.ALREADY_EXISTS(dn)
        self.db[key] = {k: _internal_attr(k, v) for k, v in modlist}
        self.db.sync()

    def delete_s(self, dn):
        """Remove the ldap object at specified dn."""
        return self.delete_ext_s(dn, serverctrls=[])

    def _getChildren(self, dn):
        return [k for k, v in self.db.items() if re.match('%s.*,%s' % (re.escape(self.__prefix), re.escape(dn)), k)]

    def delete_ext_s(self, dn, serverctrls, clientctrls=None):
        """Remove the ldap object at specified dn."""
        if server_fail:
            raise ldap.SERVER_DOWN
        try:
            key = self.key(dn)
            LOG.debug('FakeLdap delete item: dn=%s', dn)
            del self.db[key]
        except KeyError:
            LOG.debug('delete item failed: dn=%s not found.', dn)
            raise ldap.NO_SUCH_OBJECT
        self.db.sync()

    def modify_s(self, dn, modlist):
        """Modify the object at dn using the attribute list.

        :param dn: an LDAP DN
        :param modlist: a list of tuples in the following form:
                      ([MOD_ADD | MOD_DELETE | MOD_REPACE], attribute, value)
        """
        if server_fail:
            raise ldap.SERVER_DOWN
        key = self.key(dn)
        LOG.debug('modify item: dn=%(dn)s attrs=%(attrs)s', {'dn': dn, 'attrs': modlist})
        try:
            entry = self.db[key]
        except KeyError:
            LOG.debug('modify item failed: dn=%s not found.', dn)
            raise ldap.NO_SUCH_OBJECT
        for cmd, k, v in modlist:
            values = entry.setdefault(k, [])
            if cmd == ldap.MOD_ADD:
                v = _internal_attr(k, v)
                for x in v:
                    if x in values:
                        raise ldap.TYPE_OR_VALUE_EXISTS
                values += v
            elif cmd == ldap.MOD_REPLACE:
                values[:] = _internal_attr(k, v)
            elif cmd == ldap.MOD_DELETE:
                if v is None:
                    if not values:
                        LOG.debug('modify item failed: item has no attribute "%s" to delete', k)
                        raise ldap.NO_SUCH_ATTRIBUTE
                    values[:] = []
                else:
                    for val in _internal_attr(k, v):
                        try:
                            values.remove(val)
                        except ValueError:
                            LOG.debug('modify item failed: item has no attribute "%(k)s" with value "%(v)s" to delete', {'k': k, 'v': val})
                            raise ldap.NO_SUCH_ATTRIBUTE
            else:
                LOG.debug('modify item failed: unknown command %s', cmd)
                raise NotImplementedError('modify_s action %s not implemented' % cmd)
        self.db[key] = entry
        self.db.sync()

    def search_s(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0):
        """Search for all matching objects under base using the query.

        Args:
        base -- dn to search under
        scope -- search scope (base, subtree, onelevel)
        filterstr -- filter objects by
        attrlist -- attrs to return. Returns all attrs if not specified

        """
        if server_fail:
            raise ldap.SERVER_DOWN
        if not filterstr and scope != ldap.SCOPE_BASE:
            raise AssertionError('Search without filter on onelevel or subtree scope')
        if scope == ldap.SCOPE_BASE:
            try:
                item_dict = self.db[self.key(base)]
            except KeyError:
                LOG.debug('search fail: dn not found for SCOPE_BASE')
                raise ldap.NO_SUCH_OBJECT
            results = [(base, item_dict)]
        elif scope == ldap.SCOPE_SUBTREE:
            try:
                item_dict = self.db[self.key(base)]
            except KeyError:
                LOG.debug('search fail: dn not found for SCOPE_SUBTREE')
                raise ldap.NO_SUCH_OBJECT
            results = [(base, item_dict)]
            extraresults = [(k[len(self.__prefix):], v) for k, v in self.db.items() if re.match('%s.*,%s' % (re.escape(self.__prefix), re.escape(base)), k)]
            results.extend(extraresults)
        elif scope == ldap.SCOPE_ONELEVEL:

            def get_entries():
                base_dn = ldap.dn.str2dn(base)
                base_len = len(base_dn)
                for k, v in self.db.items():
                    if not k.startswith(self.__prefix):
                        continue
                    k_dn_str = k[len(self.__prefix):]
                    k_dn = ldap.dn.str2dn(k_dn_str)
                    if len(k_dn) != base_len + 1:
                        continue
                    if k_dn[-base_len:] != base_dn:
                        continue
                    yield (k_dn_str, v)
            results = list(get_entries())
        else:
            raise ldap.PROTOCOL_ERROR
        objects = []
        for dn, attrs in results:
            id_attr, id_val, _ = ldap.dn.str2dn(dn)[0][0]
            match_attrs = attrs.copy()
            match_attrs[id_attr] = [id_val]
            attrs_checked = set()
            if not filterstr or _match_query(filterstr, match_attrs, attrs_checked):
                if filterstr and scope != ldap.SCOPE_BASE and ('objectclass' not in attrs_checked):
                    raise AssertionError('No objectClass in search filter')
                attrs = {k: v for k, v in attrs.items() if not attrlist or k in attrlist}
                objects.append((dn, attrs))
        return objects

    def set_option(self, option, invalue):
        self._ldap_options[option] = invalue

    def get_option(self, option):
        value = self._ldap_options.get(option)
        return value

    def search_ext(self, base, scope, filterstr='(objectClass=*)', attrlist=None, attrsonly=0, serverctrls=None, clientctrls=None, timeout=-1, sizelimit=0):
        if clientctrls is not None or timeout != -1 or sizelimit != 0:
            raise exception.NotImplemented()
        if serverctrls and len(serverctrls) > 1:
            raise exception.NotImplemented()
        msgid = random.randint(0, 1000)
        PendingRequests[msgid] = (base, scope, filterstr, attrlist, attrsonly, serverctrls)
        return msgid

    def result3(self, msgid=ldap.RES_ANY, all=1, timeout=None, resp_ctrl_classes=None):
        """Execute async request.

        Only msgid param is supported. Request info is fetched from global
        variable `PendingRequests` by msgid, executed using search_s and
        limited if requested.
        """
        if all != 1 or timeout is not None or resp_ctrl_classes is not None:
            raise exception.NotImplemented()
        params = PendingRequests[msgid]
        results = self.search_s(*params[:5])
        serverctrls = params[5]
        ctrl = serverctrls[0]
        if ctrl.size:
            rdata = results[:ctrl.size]
        else:
            rdata = results
        rtype = None
        rmsgid = None
        serverctrls = None
        return (rtype, rdata, rmsgid, serverctrls)