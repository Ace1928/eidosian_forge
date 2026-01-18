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
class ImageFactoryProxy(glance.domain.proxy.ImageFactory):

    def __init__(self, factory, context, db_api, store_utils):
        proxy_kwargs = {'context': context, 'db_api': db_api, 'store_utils': store_utils}
        super(ImageFactoryProxy, self).__init__(factory, proxy_class=ImageProxy, proxy_kwargs=proxy_kwargs)

    def new_image(self, **kwargs):
        tags = kwargs.pop('tags', set([]))
        _enforce_image_tag_quota(tags)
        return super(ImageFactoryProxy, self).new_image(tags=tags, **kwargs)