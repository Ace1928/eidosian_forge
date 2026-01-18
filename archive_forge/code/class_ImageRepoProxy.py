import copy
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
import glance.api.common
import glance.common.exception as exception
from glance.common import utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _, _LI
class ImageRepoProxy(glance.domain.proxy.Repo):

    def __init__(self, image_repo, context, db_api, store_utils):
        self.image_repo = image_repo
        self.db_api = db_api
        proxy_kwargs = {'context': context, 'db_api': db_api, 'store_utils': store_utils}
        super(ImageRepoProxy, self).__init__(image_repo, item_proxy_class=ImageProxy, item_proxy_kwargs=proxy_kwargs)

    def _enforce_image_property_quota(self, properties):
        if CONF.image_property_quota < 0:
            return
        attempted = len([x for x in properties.keys() if not x.startswith(glance.api.common.GLANCE_RESERVED_NS)])
        maximum = CONF.image_property_quota
        if attempted > maximum:
            kwargs = {'attempted': attempted, 'maximum': maximum}
            exc = exception.ImagePropertyLimitExceeded(**kwargs)
            LOG.debug(encodeutils.exception_to_unicode(exc))
            raise exc

    def save(self, image, from_state=None):
        if image.added_new_properties():
            self._enforce_image_property_quota(image.extra_properties)
        return super(ImageRepoProxy, self).save(image, from_state=from_state)

    def add(self, image):
        self._enforce_image_property_quota(image.extra_properties)
        return super(ImageRepoProxy, self).add(image)