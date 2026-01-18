import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
class RSAContainer(RSAContainerFormatter, Container):
    _required_secrets = ['public_key', 'private_key']
    _optional_secrets = ['private_key_passphrase']
    _type = 'rsa'

    def __init__(self, api, name=None, public_key=None, private_key=None, private_key_passphrase=None, consumers=[], container_ref=None, created=None, updated=None, status=None, public_key_ref=None, private_key_ref=None, private_key_passphrase_ref=None):
        secret_refs = {}
        if public_key_ref:
            secret_refs['public_key'] = public_key_ref
        if private_key_ref:
            secret_refs['private_key'] = private_key_ref
        if private_key_passphrase_ref:
            secret_refs['private_key_passphrase'] = private_key_passphrase_ref
        super(RSAContainer, self).__init__(api=api, name=name, consumers=consumers, container_ref=container_ref, created=created, updated=updated, status=status, secret_refs=secret_refs)
        if public_key:
            self.public_key = public_key
        if private_key:
            self.private_key = private_key
        if private_key_passphrase:
            self.private_key_passphrase = private_key_passphrase

    @property
    def public_key(self):
        """Secret containing the Public Key"""
        return self._get_named_secret('public_key')

    @property
    def private_key(self):
        """Secret containing the Private Key"""
        return self._get_named_secret('private_key')

    @property
    def private_key_passphrase(self):
        """Secret containing the Passphrase"""
        return self._get_named_secret('private_key_passphrase')

    @public_key.setter
    @_immutable_after_save
    def public_key(self, value):
        super(RSAContainer, self).remove('public_key')
        super(RSAContainer, self).add('public_key', value)

    @private_key.setter
    @_immutable_after_save
    def private_key(self, value):
        super(RSAContainer, self).remove('private_key')
        super(RSAContainer, self).add('private_key', value)

    @private_key_passphrase.setter
    @_immutable_after_save
    def private_key_passphrase(self, value):
        super(RSAContainer, self).remove('private_key_passphrase')
        super(RSAContainer, self).add('private_key_passphrase', value)

    def add(self, name, sec):
        raise NotImplementedError('`add()` is not implemented for Typed Containers')

    def __repr__(self):
        return 'RSAContainer(name="{0}")'.format(self.name)