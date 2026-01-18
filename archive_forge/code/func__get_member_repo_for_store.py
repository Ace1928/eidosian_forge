from collections import abc
import copy
import functools
from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from glance.common import exception
from glance.common import format_inspector
from glance.common import store_utils
from glance.common import utils
import glance.domain.proxy
from glance.i18n import _, _LE, _LI, _LW
def _get_member_repo_for_store(image, context, db_api, store_api):
    image_member_repo = glance.db.ImageMemberRepo(context, db_api, image)
    store_image_repo = glance.location.ImageMemberRepoProxy(image_member_repo, image, context, store_api)
    return store_image_repo