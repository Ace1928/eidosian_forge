import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class KeyOrder(Order, KeyOrderFormatter):
    """KeyOrders can be used to request random key material from Barbican"""
    _type = 'key'
    _validMeta = ('name', 'algorithm', 'mode', 'bit_length', 'expiration', 'payload_content_type')

    def __init__(self, api, name=None, algorithm=None, bit_length=None, mode=None, expiration=None, payload_content_type=None, status=None, created=None, updated=None, order_ref=None, secret_ref=None, error_status_code=None, error_reason=None, sub_status=None, sub_status_message=None, creator_id=None):
        super(KeyOrder, self).__init__(api, self._type, status=status, created=created, updated=updated, meta={'name': name, 'algorithm': algorithm, 'bit_length': bit_length, 'expiration': expiration, 'payload_content_type': payload_content_type}, order_ref=order_ref, error_status_code=error_status_code, error_reason=error_reason, sub_status=sub_status, sub_status_message=sub_status_message, creator_id=creator_id)
        self._secret_ref = secret_ref
        if mode:
            self._meta['mode'] = mode

    @property
    def mode(self):
        """Encryption mode being used with this key

        The mode could be set to "CBC" for example, when requesting a key that
        will be used for AES encryption in CBC mode.
        """
        return self._meta.get('mode')

    @property
    def secret_ref(self):
        return self._secret_ref

    @mode.setter
    @immutable_after_save
    def mode(self, value):
        self._meta['mode'] = value

    def __repr__(self):
        return 'KeyOrder(order_ref={0})'.format(self.order_ref)