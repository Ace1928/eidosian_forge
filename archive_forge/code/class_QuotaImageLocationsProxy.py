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
class QuotaImageLocationsProxy(object):

    def __init__(self, image, context, db_api):
        self.image = image
        self.context = context
        self.db_api = db_api
        self.locations = image.locations

    def __cast__(self, *args, **kwargs):
        return self.locations.__cast__(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self.locations.__contains__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.locations.__delitem__(*args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        return self.locations.__delslice__(*args, **kwargs)

    def __eq__(self, other):
        return self.locations == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, *args, **kwargs):
        return self.locations.__getitem__(*args, **kwargs)

    def __iadd__(self, other):
        if not hasattr(other, '__iter__'):
            raise TypeError()
        self._check_user_storage_quota(other)
        return self.locations.__iadd__(other)

    def __iter__(self, *args, **kwargs):
        return self.locations.__iter__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.locations.__len__(*args, **kwargs)

    def __setitem__(self, key, value):
        return self.locations.__setitem__(key, value)

    def count(self, *args, **kwargs):
        return self.locations.count(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.locations.index(*args, **kwargs)

    def pop(self, *args, **kwargs):
        return self.locations.pop(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return self.locations.remove(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self.locations.reverse(*args, **kwargs)

    def _check_user_storage_quota(self, locations):
        required_size = _calc_required_size(self.context, self.image, locations)
        glance.api.common.check_quota(self.context, required_size, self.db_api)
        _enforce_image_location_quota(self.image, locations)

    def __copy__(self):
        return type(self)(self.image, self.context, self.db_api)

    def __deepcopy__(self, memo):
        self.image.locations = copy.deepcopy(self.locations, memo)
        return type(self)(self.image, self.context, self.db_api)

    def append(self, object):
        self._check_user_storage_quota([object])
        return self.locations.append(object)

    def insert(self, index, object):
        self._check_user_storage_quota([object])
        return self.locations.insert(index, object)

    def extend(self, iter):
        self._check_user_storage_quota(iter)
        return self.locations.extend(iter)