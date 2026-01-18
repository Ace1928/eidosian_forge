import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
class Replies(pyngus.ReceiverEventHandler):
    """This is the receiving link for all RPC reply messages.  Messages are
    routed to the proper incoming queue using the correlation-id header in the
    message.
    """

    def __init__(self, connection, on_ready, on_down, capacity):
        self._correlation = {}
        self._on_ready = on_ready
        self._on_down = on_down
        rname = 'openstack.org/om/receiver/[rpc-response]/%s' % uuid.uuid4().hex
        self._receiver = connection.create_receiver('rpc-response', event_handler=self, name=rname)
        self._capacity = capacity
        self._capacity_low = (capacity + 1) / 2
        self._receiver.open()

    def detach(self):
        if self._receiver:
            self._receiver.close()

    def destroy(self):
        self._correlation.clear()
        if self._receiver:
            self._receiver.destroy()
            self._receiver = None

    def prepare_for_response(self, request, callback):
        """Apply a unique message identifier to this request message. This will
        be used to identify messages received in reply.  The identifier is
        placed in the 'id' field of the request message.  It is expected that
        the identifier will appear in the 'correlation-id' field of the
        corresponding response message.

        When the caller is done receiving replies, it must call cancel_response
        """
        request.id = uuid.uuid4().hex
        self._correlation[request.id] = callback
        request.reply_to = self._receiver.source_address
        return request.id

    def cancel_response(self, msg_id):
        """Abort waiting for the response message corresponding to msg_id.
        This can be used if the request fails and no reply is expected.
        """
        try:
            del self._correlation[msg_id]
        except KeyError:
            pass

    @property
    def active(self):
        return self._receiver and self._receiver.active

    def receiver_active(self, receiver_link):
        """This is a Pyngus callback, invoked by Pyngus when the receiver_link
        has transitioned to the open state and is able to receive incoming
        messages.
        """
        LOG.debug('Replies link active src=%s', self._receiver.source_address)
        receiver_link.add_capacity(self._capacity)
        self._on_ready()

    def receiver_remote_closed(self, receiver, pn_condition):
        """This is a Pyngus callback, invoked by Pyngus when the peer of this
        receiver link has initiated closing the connection.
        """
        if pn_condition:
            LOG.error('Reply subscription closed by peer: %s', pn_condition)
        receiver.close()

    def receiver_failed(self, receiver_link, error):
        """Protocol error occurred."""
        LOG.error('Link to reply queue failed. error=%(error)s', {'error': error})
        self._on_down()

    def receiver_closed(self, receiver_link):
        self._on_down()

    def message_received(self, receiver, message, handle):
        """This is a Pyngus callback, invoked by Pyngus when a new message
        arrives on this receiver link from the peer.
        """
        key = message.correlation_id
        try:
            self._correlation[key](message)
            receiver.message_accepted(handle)
        except KeyError:
            LOG.warning("Can't find receiver for response msg id=%s, dropping!", key)
            receiver.message_modified(handle, True, True, None)
        if receiver.capacity <= self._capacity_low:
            receiver.add_capacity(self._capacity - receiver.capacity)