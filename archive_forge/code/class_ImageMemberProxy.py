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
class ImageMemberProxy(glance.domain.proxy.ImageMember):

    def __init__(self, image_member, context, db_api, store_utils):
        self.image_member = image_member
        self.context = context
        self.db_api = db_api
        self.store_utils = store_utils
        super(ImageMemberProxy, self).__init__(image_member)