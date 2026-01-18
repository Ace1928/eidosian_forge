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
class BaseLdap(object):
    DEFAULT_OU = None
    DEFAULT_STRUCTURAL_CLASSES = None
    DEFAULT_ID_ATTR = 'cn'
    DEFAULT_OBJECTCLASS = None
    DEFAULT_FILTER = None
    DEFAULT_EXTRA_ATTR_MAPPING = []
    NotFound = None
    notfound_arg = None
    options_name = None
    model = None
    attribute_options_names = {}
    immutable_attrs = []
    attribute_ignore = []
    tree_dn = None

    def __init__(self, conf):
        if conf.ldap.randomize_urls:
            urls = re.split('[\\s,]+', conf.ldap.url)
            random.shuffle(urls)
            self.LDAP_URL = ','.join(urls)
        else:
            self.LDAP_URL = conf.ldap.url
        self.LDAP_USER = conf.ldap.user
        self.LDAP_PASSWORD = conf.ldap.password
        self.LDAP_SCOPE = ldap_scope(conf.ldap.query_scope)
        self.alias_dereferencing = parse_deref(conf.ldap.alias_dereferencing)
        self.page_size = conf.ldap.page_size
        self.use_tls = conf.ldap.use_tls
        self.tls_cacertfile = conf.ldap.tls_cacertfile
        self.tls_cacertdir = conf.ldap.tls_cacertdir
        self.tls_req_cert = parse_tls_cert(conf.ldap.tls_req_cert)
        self.attribute_mapping = {}
        self.chase_referrals = conf.ldap.chase_referrals
        self.debug_level = conf.ldap.debug_level
        self.conn_timeout = conf.ldap.connection_timeout
        self.use_pool = conf.ldap.use_pool
        self.pool_size = conf.ldap.pool_size
        self.pool_retry_max = conf.ldap.pool_retry_max
        self.pool_retry_delay = conf.ldap.pool_retry_delay
        self.pool_conn_timeout = conf.ldap.pool_connection_timeout
        self.pool_conn_lifetime = conf.ldap.pool_connection_lifetime
        self.use_auth_pool = self.use_pool and conf.ldap.use_auth_pool
        self.auth_pool_size = conf.ldap.auth_pool_size
        self.auth_pool_conn_lifetime = conf.ldap.auth_pool_connection_lifetime
        if self.options_name is not None:
            self.tree_dn = getattr(conf.ldap, '%s_tree_dn' % self.options_name) or '%s,%s' % (self.DEFAULT_OU, conf.ldap.suffix)
            idatt = '%s_id_attribute' % self.options_name
            self.id_attr = getattr(conf.ldap, idatt) or self.DEFAULT_ID_ATTR
            objclass = '%s_objectclass' % self.options_name
            self.object_class = getattr(conf.ldap, objclass) or self.DEFAULT_OBJECTCLASS
            for k, v in self.attribute_options_names.items():
                v = '%s_%s_attribute' % (self.options_name, v)
                self.attribute_mapping[k] = getattr(conf.ldap, v)
            attr_mapping_opt = '%s_additional_attribute_mapping' % self.options_name
            attr_mapping = getattr(conf.ldap, attr_mapping_opt) or self.DEFAULT_EXTRA_ATTR_MAPPING
            self.extra_attr_mapping = self._parse_extra_attrs(attr_mapping)
            ldap_filter = '%s_filter' % self.options_name
            self.ldap_filter = getattr(conf.ldap, ldap_filter) or self.DEFAULT_FILTER
            member_attribute = '%s_member_attribute' % self.options_name
            self.member_attribute = getattr(conf.ldap, member_attribute, None)
            self.structural_classes = self.DEFAULT_STRUCTURAL_CLASSES
            if self.notfound_arg is None:
                self.notfound_arg = self.options_name + '_id'
            attribute_ignore = '%s_attribute_ignore' % self.options_name
            self.attribute_ignore = getattr(conf.ldap, attribute_ignore)

    def _not_found(self, object_id):
        if self.NotFound is None:
            return exception.NotFound(target=object_id)
        else:
            return self.NotFound(**{self.notfound_arg: object_id})

    @staticmethod
    def _parse_extra_attrs(option_list):
        mapping = {}
        for item in option_list:
            try:
                ldap_attr, attr_map = item.split(':')
            except ValueError:
                LOG.warning('Invalid additional attribute mapping: "%s". Format must be <ldap_attribute>:<keystone_attribute>', item)
                continue
            mapping[ldap_attr] = attr_map
        return mapping

    def get_connection(self, user=None, password=None, end_user_auth=False):
        use_pool = self.use_pool
        pool_size = self.pool_size
        pool_conn_lifetime = self.pool_conn_lifetime
        if end_user_auth:
            if not self.use_auth_pool:
                use_pool = False
            else:
                pool_size = self.auth_pool_size
                pool_conn_lifetime = self.auth_pool_conn_lifetime
        conn = _get_connection(self.LDAP_URL, use_pool, use_auth_pool=end_user_auth)
        conn = KeystoneLDAPHandler(conn=conn)
        try:
            conn.connect(self.LDAP_URL, page_size=self.page_size, alias_dereferencing=self.alias_dereferencing, use_tls=self.use_tls, tls_cacertfile=self.tls_cacertfile, tls_cacertdir=self.tls_cacertdir, tls_req_cert=self.tls_req_cert, chase_referrals=self.chase_referrals, debug_level=self.debug_level, conn_timeout=self.conn_timeout, use_pool=use_pool, pool_size=pool_size, pool_retry_max=self.pool_retry_max, pool_retry_delay=self.pool_retry_delay, pool_conn_timeout=self.pool_conn_timeout, pool_conn_lifetime=pool_conn_lifetime)
            if user is None:
                user = self.LDAP_USER
            if password is None:
                password = self.LDAP_PASSWORD
            if user and password:
                conn.simple_bind_s(user, password)
            else:
                conn.simple_bind_s()
            return conn
        except ldap.INVALID_CREDENTIALS:
            raise exception.LDAPInvalidCredentialsError()
        except ldap.SERVER_DOWN:
            raise exception.LDAPServerConnectionError(url=self.LDAP_URL)

    def _id_to_dn_string(self, object_id):
        return u'%s=%s,%s' % (self.id_attr, ldap.dn.escape_dn_chars(str(object_id)), self.tree_dn)

    def _id_to_dn(self, object_id):
        if self.LDAP_SCOPE == ldap.SCOPE_ONELEVEL:
            return self._id_to_dn_string(object_id)
        with self.get_connection() as conn:
            search_result = conn.search_s(self.tree_dn, self.LDAP_SCOPE, u'(&(%(id_attr)s=%(id)s)(objectclass=%(objclass)s))' % {'id_attr': self.id_attr, 'id': ldap.filter.escape_filter_chars(str(object_id)), 'objclass': self.object_class}, attrlist=DN_ONLY)
        if search_result:
            dn, attrs = search_result[0]
            return dn
        else:
            return self._id_to_dn_string(object_id)

    def _dn_to_id(self, dn):
        if self.id_attr == ldap.dn.str2dn(dn)[0][0][0].lower():
            return ldap.dn.str2dn(dn)[0][0][1]
        else:
            with self.get_connection() as conn:
                search_result = conn.search_s(dn, ldap.SCOPE_BASE)
            if search_result:
                try:
                    id_list = search_result[0][1][self.id_attr]
                except KeyError:
                    message = 'ID attribute %(id_attr)s not found in LDAP object %(dn)s.' % {'id_attr': self.id_attr, 'dn': search_result}
                    LOG.warning(message)
                    raise exception.NotFound(message=message)
                if len(id_list) > 1:
                    message = 'In order to keep backward compatibility, in the case of multivalued ids, we are returning the first id %(id_attr)s in the DN.' % {'id_attr': id_list[0]}
                    LOG.warning(message)
                return id_list[0]
            else:
                message = _('DN attribute %(dn)s not found in LDAP') % {'dn': dn}
                raise exception.NotFound(message=message)

    def _ldap_res_to_model(self, res):
        lower_res = {k.lower(): v for k, v in res[1].items()}
        id_attrs = lower_res.get(self.id_attr.lower())
        if not id_attrs:
            message = _('ID attribute %(id_attr)s not found in LDAP object %(dn)s') % {'id_attr': self.id_attr, 'dn': res[0]}
            raise exception.NotFound(message=message)
        if len(id_attrs) > 1:
            message = 'ID attribute %(id_attr)s for LDAP object %(dn)s has multiple values and therefore cannot be used as an ID. Will get the ID from DN instead' % {'id_attr': self.id_attr, 'dn': res[0]}
            LOG.warning(message)
            id_val = self._dn_to_id(res[0])
        else:
            id_val = id_attrs[0]
        obj = self.model(id=id_val)
        for k in obj.known_keys:
            if k in self.attribute_ignore:
                continue
            try:
                map_attr = self.attribute_mapping.get(k, k)
                if map_attr is None:
                    continue
                v = lower_res[map_attr.lower()]
            except KeyError:
                pass
            else:
                try:
                    value = v[0]
                except IndexError:
                    value = None
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8')
                    except UnicodeDecodeError:
                        LOG.error('Error decoding value %r (object id %r).', value, res[0])
                        raise
                obj[k] = value
        return obj

    def affirm_unique(self, values):
        if values.get('name') is not None:
            try:
                self.get_by_name(values['name'])
            except exception.NotFound:
                pass
            else:
                raise exception.Conflict(type=self.options_name, details=_('Duplicate name, %s.') % values['name'])
        if values.get('id') is not None:
            try:
                self.get(values['id'])
            except exception.NotFound:
                pass
            else:
                raise exception.Conflict(type=self.options_name, details=_('Duplicate ID, %s.') % values['id'])

    def create(self, values):
        self.affirm_unique(values)
        object_classes = self.structural_classes + [self.object_class]
        attrs = [('objectClass', object_classes)]
        for k, v in values.items():
            if k in self.attribute_ignore:
                continue
            if k == 'id':
                attrs.append((self.id_attr, [v]))
            elif v is not None:
                attr_type = self.attribute_mapping.get(k, k)
                if attr_type is not None:
                    attrs.append((attr_type, [v]))
                extra_attrs = [attr for attr, name in self.extra_attr_mapping.items() if name == k]
                for attr in extra_attrs:
                    attrs.append((attr, [v]))
        with self.get_connection() as conn:
            conn.add_s(self._id_to_dn(values['id']), attrs)
        return values

    def _filter_ldap_result_by_attr(self, ldap_result, ldap_attr_name):
        attr = self.attribute_mapping[ldap_attr_name]
        if not attr:
            attr_name = '%s_%s_attribute' % (self.options_name, self.attribute_options_names[ldap_attr_name])
            raise ValueError('"%(attr)s" is not a valid value for "%(attr_name)s"' % {'attr': attr, 'attr_name': attr_name})
        result = []
        for obj in ldap_result:
            ldap_res_low_keys_dict = {k.lower(): v for k, v in obj[1].items()}
            result_attr_vals = ldap_res_low_keys_dict.get(attr.lower())
            if result_attr_vals:
                if result_attr_vals[0] and result_attr_vals[0].strip():
                    result.append(obj)
        return result

    def _ldap_get(self, object_id, ldap_filter=None):
        query = u'(&(%(id_attr)s=%(id)s)%(filter)s(objectClass=%(object_class)s))' % {'id_attr': self.id_attr, 'id': ldap.filter.escape_filter_chars(str(object_id)), 'filter': ldap_filter or self.ldap_filter or '', 'object_class': self.object_class}
        with self.get_connection() as conn:
            try:
                attrs = list(set([self.id_attr] + list(self.attribute_mapping.values()) + list(self.extra_attr_mapping.keys())))
                res = conn.search_s(self.tree_dn, self.LDAP_SCOPE, query, attrs)
            except ldap.NO_SUCH_OBJECT:
                return None
        try:
            return self._filter_ldap_result_by_attr(res[:1], 'name')[0]
        except IndexError:
            return None

    def _ldap_get_limited(self, base, scope, filterstr, attrlist, sizelimit):
        with self.get_connection() as conn:
            try:
                control = ldap.controls.libldap.SimplePagedResultsControl(criticality=True, size=sizelimit, cookie='')
                msgid = conn.search_ext(base, scope, filterstr, attrlist, serverctrls=[control])
                rdata = conn.result3(msgid)
                return rdata
            except ldap.NO_SUCH_OBJECT:
                return []

    @driver_hints.truncated
    def _ldap_get_all(self, hints, ldap_filter=None):
        query = u'(&%s(objectClass=%s)(%s=*))' % (ldap_filter or self.ldap_filter or '', self.object_class, self.id_attr)
        sizelimit = 0
        attrs = list(set([self.id_attr] + list(self.attribute_mapping.values()) + list(self.extra_attr_mapping.keys())))
        if hints.limit:
            sizelimit = hints.limit['limit']
            res = self._ldap_get_limited(self.tree_dn, self.LDAP_SCOPE, query, attrs, sizelimit)
        else:
            with self.get_connection() as conn:
                try:
                    res = conn.search_s(self.tree_dn, self.LDAP_SCOPE, query, attrs)
                except ldap.NO_SUCH_OBJECT:
                    return []
        return self._filter_ldap_result_by_attr(res, 'name')

    def _ldap_get_list(self, search_base, scope, query_params=None, attrlist=None):
        query = u'(objectClass=%s)' % self.object_class
        if query_params:

            def calc_filter(attrname, value):
                val_esc = ldap.filter.escape_filter_chars(value)
                return '(%s=%s)' % (attrname, val_esc)
            query = u'(&%s%s)' % (query, ''.join([calc_filter(k, v) for k, v in query_params.items()]))
        with self.get_connection() as conn:
            return conn.search_s(search_base, scope, query, attrlist)

    def get(self, object_id, ldap_filter=None):
        res = self._ldap_get(object_id, ldap_filter)
        if res is None:
            raise self._not_found(object_id)
        else:
            return self._ldap_res_to_model(res)

    def get_by_name(self, name, ldap_filter=None):
        query = u'(%s=%s)' % (self.attribute_mapping['name'], ldap.filter.escape_filter_chars(str(name)))
        res = self.get_all(query)
        try:
            return res[0]
        except IndexError:
            raise self._not_found(name)

    def get_all(self, ldap_filter=None, hints=None):
        hints = hints or driver_hints.Hints()
        return [self._ldap_res_to_model(x) for x in self._ldap_get_all(hints, ldap_filter)]

    def update(self, object_id, values, old_obj=None):
        if old_obj is None:
            old_obj = self.get(object_id)
        modlist = []
        for k, v in values.items():
            if k == 'id':
                continue
            if k in self.attribute_ignore:
                if k == 'enabled' and (not v):
                    action = _("Disabling an entity where the 'enable' attribute is ignored by configuration.")
                    raise exception.ForbiddenAction(action=action)
                continue
            if k in old_obj and old_obj[k] == v:
                continue
            if k in self.immutable_attrs:
                msg = _('Cannot change %(option_name)s %(attr)s') % {'option_name': self.options_name, 'attr': k}
                raise exception.ValidationError(msg)
            if v is None:
                if old_obj.get(k) is not None:
                    modlist.append((ldap.MOD_DELETE, self.attribute_mapping.get(k, k), None))
                continue
            current_value = old_obj.get(k)
            if current_value is None:
                op = ldap.MOD_ADD
                modlist.append((op, self.attribute_mapping.get(k, k), [v]))
            elif current_value != v:
                op = ldap.MOD_REPLACE
                modlist.append((op, self.attribute_mapping.get(k, k), [v]))
        if modlist:
            with self.get_connection() as conn:
                try:
                    conn.modify_s(self._id_to_dn(object_id), modlist)
                except ldap.NO_SUCH_OBJECT:
                    raise self._not_found(object_id)
        return self.get(object_id)

    def add_member(self, member_dn, member_list_dn):
        """Add member to the member list.

        :param member_dn: DN of member to be added.
        :param member_list_dn: DN of group to which the
                               member will be added.

        :raises keystone.exception.Conflict: If the user was already a member.
        :raises self.NotFound: If the group entry didn't exist.
        """
        with self.get_connection() as conn:
            try:
                mod = (ldap.MOD_ADD, self.member_attribute, member_dn)
                conn.modify_s(member_list_dn, [mod])
            except ldap.TYPE_OR_VALUE_EXISTS:
                raise exception.Conflict(_('Member %(member)s is already a member of group %(group)s') % {'member': member_dn, 'group': member_list_dn})
            except ldap.NO_SUCH_OBJECT:
                raise self._not_found(member_list_dn)

    def filter_query(self, hints, query=None):
        """Apply filtering to a query.

        :param hints: contains the list of filters, which may be None,
                      indicating that there are no filters to be applied.
                      If it's not None, then any filters satisfied here will be
                      removed so that the caller will know if any filters
                      remain to be applied.
        :param query: LDAP query into which to include filters

        :returns query: LDAP query, updated with any filters satisfied

        """

        def build_filter(filter_):
            """Build a filter for the query.

            :param filter_: the dict that describes this filter

            :returns query: LDAP query term to be added

            """
            ldap_attr = self.attribute_mapping[filter_['name']]
            val_esc = ldap.filter.escape_filter_chars(filter_['value'])
            if filter_['case_sensitive']:
                return
            if filter_['name'] == 'enabled':
                return
            if filter_['comparator'] == 'equals':
                query_term = u'(%(attr)s=%(val)s)' % {'attr': ldap_attr, 'val': val_esc}
            elif filter_['comparator'] == 'contains':
                query_term = u'(%(attr)s=*%(val)s*)' % {'attr': ldap_attr, 'val': val_esc}
            elif filter_['comparator'] == 'startswith':
                query_term = u'(%(attr)s=%(val)s*)' % {'attr': ldap_attr, 'val': val_esc}
            elif filter_['comparator'] == 'endswith':
                query_term = u'(%(attr)s=*%(val)s)' % {'attr': ldap_attr, 'val': val_esc}
            else:
                return
            return query_term
        if query is None:
            query = ''
        if hints is None:
            return query
        filter_list = []
        satisfied_filters = []
        for filter_ in hints.filters:
            if filter_['name'] not in self.attribute_mapping:
                continue
            new_filter = build_filter(filter_)
            if new_filter is not None:
                filter_list.append(new_filter)
                satisfied_filters.append(filter_)
        if filter_list:
            query = u'(&%s%s)' % (query, ''.join(filter_list))
        for filter_ in satisfied_filters:
            hints.filters.remove(filter_)
        return query