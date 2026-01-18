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
class QuotaImageTagsProxy(object):

    def __init__(self, orig_set):
        if orig_set is None:
            orig_set = set([])
        self.tags = orig_set

    def add(self, item):
        self.tags.add(item)
        _enforce_image_tag_quota(self.tags)

    def __cast__(self, *args, **kwargs):
        return self.tags.__cast__(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self.tags.__contains__(*args, **kwargs)

    def __eq__(self, other):
        return self.tags == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self, *args, **kwargs):
        return self.tags.__iter__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.tags.__len__(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'tags':
            try:
                return self.__getattribute__('tags')
            except AttributeError:
                return None
        return getattr(self.tags, name)