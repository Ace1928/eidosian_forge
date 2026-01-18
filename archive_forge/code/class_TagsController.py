import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2.model.metadef_tag import MetadefTags
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
from glance.i18n import _
import glance.notifier
import glance.schema
class TagsController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None, schema=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.gateway = glance.gateway.Gateway(db_api=self.db_api, notifier=self.notifier, policy_enforcer=self.policy)
        self.schema = schema or get_schema()
        self.tag_schema_link = '/v2/schemas/metadefs/tag'

    def create(self, req, namespace, tag_name):
        tag_factory = self.gateway.get_metadef_tag_factory(req.context)
        tag_repo = self.gateway.get_metadef_tag_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        tag_name_as_dict = {'name': tag_name}
        try:
            self.schema.validate(tag_name_as_dict)
        except exception.InvalidObject as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).add_metadef_tag()
            new_meta_tag = tag_factory.new_tag(namespace=namespace, **tag_name_as_dict)
            tag_repo.add(new_meta_tag)
        except exception.Invalid as e:
            msg = _("Couldn't create metadata tag: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to create metadata tag within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return MetadefTag.to_wsme_model(new_meta_tag)

    def create_tags(self, req, metadata_tags, namespace):
        tag_factory = self.gateway.get_metadef_tag_factory(req.context)
        tag_repo = self.gateway.get_metadef_tag_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).add_metadef_tags()
            can_append = strutils.bool_from_string(req.headers.get('X-Openstack-Append'))
            tag_list = []
            for metadata_tag in metadata_tags.tags:
                tag_list.append(tag_factory.new_tag(namespace=namespace, **metadata_tag.to_dict()))
            tag_repo.add_tags(tag_list, can_append)
            tag_list_out = [MetadefTag(**{'name': db_metatag.name}) for db_metatag in tag_list]
            metadef_tags = MetadefTags()
            metadef_tags.tags = tag_list_out
        except exception.Forbidden as e:
            LOG.debug("User not permitted to create metadata tags within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return metadef_tags

    def index(self, req, namespace, marker=None, limit=None, sort_key='created_at', sort_dir='desc', filters=None):
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).get_metadef_tags()
            filters = filters or dict()
            filters['namespace'] = namespace
            tag_repo = self.gateway.get_metadef_tag_repo(req.context)
            if marker:
                metadef_tag = tag_repo.get(namespace, marker)
                marker = metadef_tag.tag_id
            db_metatag_list = tag_repo.list(marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, filters=filters)
            tag_list = [MetadefTag(**{'name': db_metatag.name}) for db_metatag in db_metatag_list]
            metadef_tags = MetadefTags()
            metadef_tags.tags = tag_list
        except exception.Forbidden as e:
            LOG.debug("User not permitted to retrieve metadata tags within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return metadef_tags

    def show(self, req, namespace, tag_name):
        meta_tag_repo = self.gateway.get_metadef_tag_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).get_metadef_tag()
            metadef_tag = meta_tag_repo.get(namespace, tag_name)
            return MetadefTag.to_wsme_model(metadef_tag)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to show metadata tag '%s' within '%s' namespace", tag_name, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def update(self, req, metadata_tag, namespace, tag_name):
        meta_repo = self.gateway.get_metadef_tag_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).modify_metadef_tag()
            metadef_tag = meta_repo.get(namespace, tag_name)
            metadef_tag._old_name = metadef_tag.name
            metadef_tag.name = wsme_utils._get_value(metadata_tag.name)
            updated_metadata_tag = meta_repo.save(metadef_tag)
        except exception.Invalid as e:
            msg = _("Couldn't update metadata tag: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to update metadata tag '%s' within '%s' namespace", tag_name, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return MetadefTag.to_wsme_model(updated_metadata_tag)

    def delete(self, req, namespace, tag_name):
        meta_repo = self.gateway.get_metadef_tag_repo(req.context)
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).delete_metadef_tag()
            metadef_tag = meta_repo.get(namespace, tag_name)
            metadef_tag.delete()
            meta_repo.remove(metadef_tag)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata tag '%s' within '%s' namespace", tag_name, namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)