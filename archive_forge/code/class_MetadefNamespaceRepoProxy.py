import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
class MetadefNamespaceRepoProxy(NotificationRepoProxy, domain_proxy.MetadefNamespaceRepo):

    def get_super_class(self):
        return domain_proxy.MetadefNamespaceRepo

    def get_proxy_class(self):
        return MetadefNamespaceProxy

    def get_payload(self, obj):
        return format_metadef_namespace_notification(obj)

    def save(self, metadef_namespace):
        name = getattr(metadef_namespace, '_old_namespace', metadef_namespace.namespace)
        result = super(MetadefNamespaceRepoProxy, self).save(metadef_namespace)
        self.send_notification('metadef_namespace.update', metadef_namespace, extra_payload={'namespace_old': name})
        return result

    def add(self, metadef_namespace):
        result = super(MetadefNamespaceRepoProxy, self).add(metadef_namespace)
        self.send_notification('metadef_namespace.create', metadef_namespace)
        return result

    def remove(self, metadef_namespace):
        result = super(MetadefNamespaceRepoProxy, self).remove(metadef_namespace)
        self.send_notification('metadef_namespace.delete', metadef_namespace, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime()})
        return result

    def remove_objects(self, metadef_namespace):
        result = super(MetadefNamespaceRepoProxy, self).remove_objects(metadef_namespace)
        self.send_notification('metadef_namespace.delete_objects', metadef_namespace)
        return result

    def remove_properties(self, metadef_namespace):
        result = super(MetadefNamespaceRepoProxy, self).remove_properties(metadef_namespace)
        self.send_notification('metadef_namespace.delete_properties', metadef_namespace)
        return result

    def remove_tags(self, metadef_namespace):
        result = super(MetadefNamespaceRepoProxy, self).remove_tags(metadef_namespace)
        self.send_notification('metadef_namespace.delete_tags', metadef_namespace)
        return result