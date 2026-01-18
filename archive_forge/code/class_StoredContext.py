import functools
from keystoneauth1 import access
from keystoneauth1.identity import access as access_plugin
from keystoneauth1.identity import generic
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import importutils
from heat.common import config
from heat.common import endpoint_utils
from heat.common import exception
from heat.common import policy
from heat.common import wsgi
from heat.engine import clients
class StoredContext(RequestContext):

    def _load_keystone_data(self):
        self._keystone_loaded = True
        auth_ref = self.auth_plugin.get_access(self.keystone_session)
        self.roles = auth_ref.role_names
        self.user_domain_id = auth_ref.user_domain_id
        self.project_domain_id = auth_ref.project_domain_id

    @property
    def roles(self):
        if self._roles is None:
            if not getattr(self, '_keystone_loaded', False):
                self._load_keystone_data()
        return self._roles

    @roles.setter
    def roles(self, roles):
        self._roles = roles

    @property
    def user_domain_id(self):
        if self._user_domain_id is None:
            if not getattr(self, '_keystone_loaded', False):
                self._load_keystone_data()
        return self._user_domain_id

    @user_domain_id.setter
    def user_domain_id(self, user_domain_id):
        self._user_domain_id = user_domain_id

    @property
    def project_domain_id(self):
        if self._project_domain_id is None:
            if not getattr(self, '_keystone_loaded', False):
                self._load_keystone_data()
        return self._project_domain_id

    @project_domain_id.setter
    def project_domain_id(self, project_domain_id):
        self._project_domain_id = project_domain_id