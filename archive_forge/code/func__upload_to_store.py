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
def _upload_to_store(self, data, verifier, store=None, size=None):
    """
        Upload data to store

        :param data: data to upload to store
        :param verifier: for signature verification
        :param store: store to upload data to
        :param size: data size
        :return:
        """
    hashing_algo = self.image.os_hash_algo or CONF['hashing_algorithm']
    if CONF.enabled_backends:
        location, size, checksum, multihash, loc_meta = self.store_api.add_with_multihash(CONF, self.image.image_id, utils.LimitingReader(utils.CooperativeReader(data), CONF.image_size_cap), size, store, hashing_algo, context=self.context, verifier=verifier)
    else:
        location, size, checksum, multihash, loc_meta = self.store_api.add_to_backend_with_multihash(CONF, self.image.image_id, utils.LimitingReader(utils.CooperativeReader(data), CONF.image_size_cap), size, hashing_algo, context=self.context, verifier=verifier)
    self._verify_signature(verifier, location, loc_meta)
    for attr, data in {'size': size, 'os_hash_value': multihash, 'checksum': checksum}.items():
        self._verify_uploaded_data(data, attr)
    self.image.locations.append({'url': location, 'metadata': loc_meta, 'status': 'active'})
    self.image.checksum = checksum
    self.image.os_hash_value = multihash
    self.image.size = size
    self.image.os_hash_algo = hashing_algo