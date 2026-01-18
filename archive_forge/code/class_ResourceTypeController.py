import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_resource_type import ResourceType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociations
from glance.api.v2.model.metadef_resource_type import ResourceTypes
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
class ResourceTypeController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.gateway = glance.gateway.Gateway(db_api=self.db_api, notifier=self.notifier, policy_enforcer=self.policy)

    def index(self, req):
        try:
            filters = {'namespace': None}
            rs_type_repo = self.gateway.get_metadef_resource_type_repo(req.context)
            api_policy.MetadefAPIPolicy(req.context, enforcer=self.policy).list_metadef_resource_types()
            db_resource_type_list = rs_type_repo.list(filters=filters)
            resource_type_list = [ResourceType.to_wsme_model(resource_type) for resource_type in db_resource_type_list]
            resource_types = ResourceTypes()
            resource_types.resource_types = resource_type_list
        except exception.Forbidden as e:
            LOG.debug('User not permitted to retrieve metadata resource types index')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return resource_types

    def show(self, req, namespace):
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).list_metadef_resource_types()
            filters = {'namespace': namespace}
            rs_type_repo = self.gateway.get_metadef_resource_type_repo(req.context)
            db_type_list = rs_type_repo.list(filters=filters)
            rs_type_list = [ResourceTypeAssociation.to_wsme_model(rs_type) for rs_type in db_type_list if api_policy.MetadefAPIPolicy(req.context, md_resource=rs_type.namespace, enforcer=self.policy).check('get_metadef_resource_type')]
            resource_types = ResourceTypeAssociations()
            resource_types.resource_type_associations = rs_type_list
        except exception.Forbidden as e:
            LOG.debug("User not permitted to retrieve metadata resource types within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return resource_types

    def create(self, req, resource_type, namespace):
        rs_type_factory = self.gateway.get_metadef_resource_type_factory(req.context)
        rs_type_repo = self.gateway.get_metadef_resource_type_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).add_metadef_resource_type_association()
            new_resource_type = rs_type_factory.new_resource_type(namespace=namespace, **resource_type.to_dict())
            rs_type_repo.add(new_resource_type)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to create metadata resource type within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return ResourceTypeAssociation.to_wsme_model(new_resource_type)

    def delete(self, req, namespace, resource_type):
        rs_type_repo = self.gateway.get_metadef_resource_type_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).remove_metadef_resource_type_association()
            filters = {}
            found = False
            filters['namespace'] = namespace
            db_resource_type_list = rs_type_repo.list(filters=filters)
            for db_resource_type in db_resource_type_list:
                if db_resource_type.name == resource_type:
                    db_resource_type.delete()
                    rs_type_repo.remove(db_resource_type)
                    found = True
            if not found:
                raise exception.NotFound()
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata resource type '%s' within '%s' namespace", resource_type, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound:
            msg = _('Failed to find resource type %(resourcetype)s to delete') % {'resourcetype': resource_type}
            LOG.error(msg)
            raise webob.exc.HTTPNotFound(explanation=msg)