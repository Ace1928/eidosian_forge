import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
import glance.api.v2.metadef_properties as properties
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_object import MetadefObjects
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
from glance.i18n import _
import glance.notifier
import glance.schema
class MetadefObjectsController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.gateway = glance.gateway.Gateway(db_api=self.db_api, notifier=self.notifier, policy_enforcer=self.policy)
        self.obj_schema_link = '/v2/schemas/metadefs/object'

    def create(self, req, metadata_object, namespace):
        object_factory = self.gateway.get_metadef_object_factory(req.context)
        object_repo = self.gateway.get_metadef_object_repo(req.context)
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
            except exception.Forbidden:
                msg = _('Namespace %s not found') % namespace
                raise exception.NotFound(msg)
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).add_metadef_object()
            new_meta_object = object_factory.new_object(namespace=namespace, **metadata_object.to_dict())
            object_repo.add(new_meta_object)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to create metadata object within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.Invalid as e:
            msg = _("Couldn't create metadata object: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return MetadefObject.to_wsme_model(new_meta_object, get_object_href(namespace, new_meta_object), self.obj_schema_link)

    def index(self, req, namespace, marker=None, limit=None, sort_key='created_at', sort_dir='desc', filters=None):
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
            except exception.Forbidden:
                msg = _('Namespace %s not found') % namespace
                raise exception.NotFound(msg)
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).get_metadef_objects()
            filters = filters or dict()
            filters['namespace'] = namespace
            object_repo = self.gateway.get_metadef_object_repo(req.context)
            db_metaobject_list = object_repo.list(marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, filters=filters)
            object_list = [MetadefObject.to_wsme_model(obj, get_object_href(namespace, obj), self.obj_schema_link) for obj in db_metaobject_list if api_policy.MetadefAPIPolicy(req.context, md_resource=obj.namespace, enforcer=self.policy).check('get_metadef_object')]
            metadef_objects = MetadefObjects()
            metadef_objects.objects = object_list
        except exception.Forbidden as e:
            LOG.debug("User not permitted to retrieve metadata objects within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return metadef_objects

    def show(self, req, namespace, object_name):
        meta_object_repo = self.gateway.get_metadef_object_repo(req.context)
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
            except exception.Forbidden:
                msg = _('Namespace %s not found') % namespace
                raise exception.NotFound(msg)
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).get_metadef_object()
            metadef_object = meta_object_repo.get(namespace, object_name)
            return MetadefObject.to_wsme_model(metadef_object, get_object_href(namespace, metadef_object), self.obj_schema_link)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to show metadata object '%s' within '%s' namespace", namespace, object_name)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def update(self, req, metadata_object, namespace, object_name):
        meta_repo = self.gateway.get_metadef_object_repo(req.context)
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
            except exception.Forbidden:
                msg = _('Namespace %s not found') % namespace
                raise exception.NotFound(msg)
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).modify_metadef_object()
            metadef_object = meta_repo.get(namespace, object_name)
            metadef_object._old_name = metadef_object.name
            metadef_object.name = wsme_utils._get_value(metadata_object.name)
            metadef_object.description = wsme_utils._get_value(metadata_object.description)
            metadef_object.required = wsme_utils._get_value(metadata_object.required)
            metadef_object.properties = wsme_utils._get_value(metadata_object.properties)
            updated_metadata_obj = meta_repo.save(metadef_object)
        except exception.Invalid as e:
            msg = _("Couldn't update metadata object: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to update metadata object '%s' within '%s' namespace ", object_name, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return MetadefObject.to_wsme_model(updated_metadata_obj, get_object_href(namespace, updated_metadata_obj), self.obj_schema_link)

    def delete(self, req, namespace, object_name):
        meta_repo = self.gateway.get_metadef_object_repo(req.context)
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
            except exception.Forbidden:
                msg = _('Namespace %s not found') % namespace
                raise exception.NotFound(msg)
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).delete_metadef_object()
            metadef_object = meta_repo.get(namespace, object_name)
            metadef_object.delete()
            meta_repo.remove(metadef_object)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata object '%s' within '%s' namespace", object_name, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)