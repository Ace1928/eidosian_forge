from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class MetadefAPIPolicy(APIPolicyBase):

    def __init__(self, context, md_resource=None, target=None, enforcer=None):
        self._context = context
        self._md_resource = md_resource
        if not target:
            self._target = self._build_target()
        else:
            self._target = target
        self.enforcer = enforcer or policy.Enforcer()
        super(MetadefAPIPolicy, self).__init__(context, target=self._target, enforcer=self.enforcer)

    def _build_target(self):
        target = {'project_id': self._context.project_id}
        if self._md_resource:
            target['project_id'] = self._md_resource.owner
            target['visibility'] = self._md_resource.visibility
        return target

    def _enforce(self, rule_name):
        """Translate Forbidden->NotFound for images."""
        try:
            super(MetadefAPIPolicy, self)._enforce(rule_name)
        except webob.exc.HTTPForbidden:
            if rule_name == 'get_metadef_namespace' or not self.check('get_metadef_namespace'):
                raise webob.exc.HTTPNotFound()
            raise

    def check(self, name, *args):
        try:
            return super(MetadefAPIPolicy, self).check(name, *args)
        except webob.exc.HTTPNotFound:
            return False

    def get_metadef_namespace(self):
        self._enforce('get_metadef_namespace')

    def get_metadef_namespaces(self):
        self._enforce('get_metadef_namespaces')

    def add_metadef_namespace(self):
        self._enforce('add_metadef_namespace')

    def modify_metadef_namespace(self):
        self._enforce('modify_metadef_namespace')

    def delete_metadef_namespace(self):
        self._enforce('delete_metadef_namespace')

    def get_metadef_objects(self):
        self._enforce('get_metadef_objects')

    def add_metadef_object(self):
        self._enforce('add_metadef_object')

    def get_metadef_object(self):
        self._enforce('get_metadef_object')

    def modify_metadef_object(self):
        self._enforce('modify_metadef_object')

    def delete_metadef_object(self):
        self._enforce('delete_metadef_object')

    def add_metadef_tag(self):
        self._enforce('add_metadef_tag')

    def get_metadef_tags(self):
        self._enforce('get_metadef_tags')

    def add_metadef_tags(self):
        self._enforce('add_metadef_tags')

    def get_metadef_tag(self):
        self._enforce('get_metadef_tag')

    def modify_metadef_tag(self):
        self._enforce('modify_metadef_tag')

    def delete_metadef_tag(self):
        self._enforce('delete_metadef_tag')

    def delete_metadef_tags(self):
        self._enforce('delete_metadef_tags')

    def add_metadef_property(self):
        self._enforce('add_metadef_property')

    def get_metadef_properties(self):
        self._enforce('get_metadef_properties')

    def remove_metadef_property(self):
        self._enforce('remove_metadef_property')

    def get_metadef_property(self):
        self._enforce('get_metadef_property')

    def modify_metadef_property(self):
        self._enforce('modify_metadef_property')

    def add_metadef_resource_type_association(self):
        self._enforce('add_metadef_resource_type_association')

    def list_metadef_resource_types(self):
        self._enforce('list_metadef_resource_types')

    def get_metadef_resource_type(self):
        self._enforce('get_metadef_resource_type')

    def remove_metadef_resource_type_association(self):
        self._enforce('remove_metadef_resource_type_association')