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
def _enforce_image_tag_quota(tags):
    if CONF.image_tag_quota < 0:
        return
    if not tags:
        return
    if len(tags) > CONF.image_tag_quota:
        raise exception.ImageTagLimitExceeded(attempted=len(tags), maximum=CONF.image_tag_quota)