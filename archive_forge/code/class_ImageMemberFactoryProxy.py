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
class ImageMemberFactoryProxy(glance.domain.proxy.ImageMembershipFactory):

    def __init__(self, member_factory, context, db_api, store_utils):
        self.db_api = db_api
        self.context = context
        proxy_kwargs = {'context': context, 'db_api': db_api, 'store_utils': store_utils}
        super(ImageMemberFactoryProxy, self).__init__(member_factory, proxy_class=ImageMemberProxy, proxy_kwargs=proxy_kwargs)

    def _enforce_image_member_quota(self, image):
        if CONF.image_member_quota < 0:
            return
        current_member_count = self.db_api.image_member_count(self.context, image.image_id)
        attempted = current_member_count + 1
        maximum = CONF.image_member_quota
        if attempted > maximum:
            raise exception.ImageMemberLimitExceeded(attempted=attempted, maximum=maximum)

    def new_image_member(self, image, member_id):
        self._enforce_image_member_quota(image)
        return super(ImageMemberFactoryProxy, self).new_image_member(image, member_id)