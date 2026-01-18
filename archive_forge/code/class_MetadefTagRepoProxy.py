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
class MetadefTagRepoProxy(NotificationRepoProxy, domain_proxy.MetadefTagRepo):

    def get_super_class(self):
        return domain_proxy.MetadefTagRepo

    def get_proxy_class(self):
        return MetadefTagProxy

    def get_payload(self, obj):
        return format_metadef_tag_notification(obj)

    def save(self, metadef_tag):
        name = getattr(metadef_tag, '_old_name', metadef_tag.name)
        result = super(MetadefTagRepoProxy, self).save(metadef_tag)
        self.send_notification('metadef_tag.update', metadef_tag, extra_payload={'namespace': metadef_tag.namespace.namespace, 'name_old': name})
        return result

    def add(self, metadef_tag):
        result = super(MetadefTagRepoProxy, self).add(metadef_tag)
        self.send_notification('metadef_tag.create', metadef_tag)
        return result

    def add_tags(self, metadef_tags, can_append=False):
        result = super(MetadefTagRepoProxy, self).add_tags(metadef_tags, can_append)
        for metadef_tag in metadef_tags:
            self.send_notification('metadef_tag.create', metadef_tag)
        return result

    def remove(self, metadef_tag):
        result = super(MetadefTagRepoProxy, self).remove(metadef_tag)
        self.send_notification('metadef_tag.delete', metadef_tag, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime(), 'namespace': metadef_tag.namespace.namespace})
        return result