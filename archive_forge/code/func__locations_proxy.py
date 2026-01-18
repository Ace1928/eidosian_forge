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
def _locations_proxy(target, attr):
    """
    Make a location property proxy on the image object.

    :param target: the image object on which to add the proxy
    :param attr: the property proxy we want to hook
    """

    def get_attr(self):
        value = getattr(getattr(self, target), attr)
        return StoreLocations(self, value)

    def set_attr(self, value):
        if not isinstance(value, (list, StoreLocations)):
            reason = _('Invalid locations')
            raise exception.BadStoreUri(message=reason)
        ori_value = getattr(getattr(self, target), attr)
        if ori_value != value:
            ordered_value = sorted([loc['url'] for loc in value])
            ordered_ori = sorted([loc['url'] for loc in ori_value])
            if len(ori_value) > 0 and ordered_value != ordered_ori:
                raise exception.Invalid(_('Original locations is not empty: %s') % ori_value)
            if ordered_value != ordered_ori:
                for loc in value:
                    _check_image_location(self.context, self.store_api, self.store_utils, loc)
                    loc['status'] = 'active'
                    if _count_duplicated_locations(value, loc) > 1:
                        raise exception.DuplicateLocation(location=loc['url'])
                _set_image_size(self.context, getattr(self, target), value)
            else:
                for loc in value:
                    loc['status'] = 'active'
            return setattr(getattr(self, target), attr, list(value))

    def del_attr(self):
        value = getattr(getattr(self, target), attr)
        while len(value):
            self.store_utils.delete_image_location_from_backend(self.context, self.image.image_id, value[0])
            del value[0]
            setattr(getattr(self, target), attr, value)
        return delattr(getattr(self, target), attr)
    return property(get_attr, set_attr, del_attr)