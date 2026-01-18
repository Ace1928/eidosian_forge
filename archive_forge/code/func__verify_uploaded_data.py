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
def _verify_uploaded_data(self, value, attribute_name):
    """
        Verify value of attribute_name uploaded data

        :param value: value to compare
        :param attribute_name: attribute name of the image to compare with
        """
    image_value = getattr(self.image, attribute_name)
    if image_value is not None and value != image_value:
        msg = _('%s of uploaded data is different from current value set on the image.')
        LOG.error(msg, attribute_name)
        raise exception.UploadException(msg % attribute_name)