import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class AsymmetricOrder(Order, AsymmetricOrderFormatter):
    _type = 'asymmetric'

    def __init__(self, api, name=None, algorithm=None, bit_length=None, mode=None, passphrase=None, pass_phrase=None, expiration=None, payload_content_type=None, status=None, created=None, updated=None, order_ref=None, container_ref=None, error_status_code=None, error_reason=None, sub_status=None, sub_status_message=None, creator_id=None):
        super(AsymmetricOrder, self).__init__(api, self._type, status=status, created=created, updated=updated, meta={'name': name, 'algorithm': algorithm, 'bit_length': bit_length, 'expiration': expiration, 'payload_content_type': payload_content_type}, order_ref=order_ref, error_status_code=error_status_code, error_reason=error_reason, sub_status=sub_status, sub_status_message=sub_status_message, creator_id=creator_id)
        self._container_ref = container_ref
        if passphrase:
            self._meta['pass_phrase'] = passphrase
        elif pass_phrase:
            self._meta['pass_phrase'] = pass_phrase

    @property
    def container_ref(self):
        return self._container_ref

    @property
    def pass_phrase(self):
        """Passphrase to be used for passphrase protected asymmetric keys"""
        return self._meta.get('pass_phrase')

    @pass_phrase.setter
    @immutable_after_save
    def pass_phrase(self, value):
        self._meta['pass_phrase'] = value

    def __repr__(self):
        return 'AsymmetricOrder(order_ref={0})'.format(self.order_ref)