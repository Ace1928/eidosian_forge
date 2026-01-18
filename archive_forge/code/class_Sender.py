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
class Sender(pyngus.SenderEventHandler):
    """A link for sending to a particular destination on the message bus.
    """

    def __init__(self, destination, scheduler, delay, service):
        super(Sender, self).__init__()
        self._destination = destination
        self._service = service
        self._address = None
        self._link = None
        self._scheduler = scheduler
        self._delay = delay
        self._pending_sends = collections.deque()
        self._unacked = set()
        self._reply_link = None
        self._connection = None
        self._resend_timer = None

    @property
    def pending_messages(self):
        return len(self._pending_sends)

    @property
    def unacked_messages(self):
        return len(self._unacked)

    def attach(self, connection, reply_link, addresser):
        """Open the link. Called by the Controller when the AMQP connection
        becomes active.
        """
        self._connection = connection
        self._reply_link = reply_link
        self._address = addresser.resolve(self._destination, self._service)
        LOG.debug('Sender %s attached', self._address)
        self._link = self._open_link()

    def detach(self):
        """Close the link.  Called by the controller when shutting down or in
        response to a close requested by the remote.  May be re-attached later
        (after a reset is done)
        """
        LOG.debug('Sender %s detached', self._address)
        self._connection = None
        self._reply_link = None
        if self._resend_timer:
            self._resend_timer.cancel()
            self._resend_timer = None
        if self._link:
            self._link.close()

    def reset(self, reason='Link reset'):
        """Called by the controller on connection failover. Release all link
        resources, abort any in-flight messages, and check the retry limit on
        all pending send requests.
        """
        self._address = None
        self._connection = None
        self._reply_link = None
        if self._link:
            self._link.destroy()
            self._link = None
        self._abort_unacked(reason)
        self._check_retry_limit(reason)

    def destroy(self, reason='Link destroyed'):
        """Destroy the sender and all pending messages.  Called on driver
        shutdown.
        """
        LOG.debug('Sender %s destroyed', self._address)
        self.reset(reason)
        self._abort_pending(reason)

    def send_message(self, send_task):
        """Send a message out the link.
        """
        if not self._can_send or self._pending_sends:
            self._pending_sends.append(send_task)
        else:
            self._send(send_task)

    def cancel_send(self, send_task):
        """Attempts to cancel a send request.  It is possible that the send has
        already completed, so this is best-effort.
        """
        self._unacked.discard(send_task)
        try:
            self._pending_sends.remove(send_task)
        except ValueError:
            pass

    def sender_active(self, sender_link):
        LOG.debug('Sender %s active', self._address)
        self._send_pending()

    def credit_granted(self, sender_link):
        pass

    def sender_remote_closed(self, sender_link, pn_condition):
        LOG.warning('Sender %(addr)s failed due to remote initiated close: condition=%(cond)s', {'addr': self._address, 'cond': pn_condition})
        self._link.close()

    def sender_closed(self, sender_link):
        self._handle_sender_closed()

    def sender_failed(self, sender_link, error):
        """Protocol error occurred."""
        LOG.warning('Sender %(addr)s failed error=%(error)s', {'addr': self._address, 'error': error})
        self._handle_sender_closed(str(error))

    def _handle_sender_closed(self, reason='Sender closed'):
        self._abort_unacked(reason)
        if self._connection:
            self._check_retry_limit(reason)
            self._scheduler.defer(self._reopen_link, self._delay)

    def _check_retry_limit(self, reason):
        expired = set()
        for send_task in self._pending_sends:
            if not send_task._can_retry:
                expired.add(send_task)
                send_task._on_error('Message send failed: %s' % reason)
        while expired:
            self._pending_sends.remove(expired.pop())

    def _abort_unacked(self, error):
        while self._unacked:
            send_task = self._unacked.pop()
            send_task._on_error('Message send failed: %s' % error)

    def _abort_pending(self, error):
        while self._pending_sends:
            send_task = self._pending_sends.popleft()
            send_task._on_error('Message send failed: %s' % error)

    @property
    def _can_send(self):
        return self._link and self._link.active
    _TIMED_OUT = pyngus.SenderLink.TIMED_OUT
    _ACCEPTED = pyngus.SenderLink.ACCEPTED
    _RELEASED = pyngus.SenderLink.RELEASED
    _MODIFIED = pyngus.SenderLink.MODIFIED

    def _send(self, send_task):
        send_task._prepare(self)
        send_task.message.address = self._address
        if send_task.wait_for_ack:
            self._unacked.add(send_task)

            def pyngus_callback(link, handle, state, info):
                if state == Sender._TIMED_OUT:
                    return
                self._unacked.discard(send_task)
                if state == Sender._ACCEPTED:
                    send_task._on_ack(Sender._ACCEPTED, info)
                elif state == Sender._RELEASED or (state == Sender._MODIFIED and (not info.get('delivery-failed')) and (not info.get('undeliverable-here'))):
                    self._resend(send_task)
                else:
                    send_task._on_ack(state, info)
            self._link.send(send_task.message, delivery_callback=pyngus_callback, handle=self, deadline=send_task.deadline)
        else:
            self._link.send(send_task.message, delivery_callback=None, handle=self, deadline=send_task.deadline)
            send_task._on_ack(pyngus.SenderLink.ACCEPTED, {})

    def _resend(self, send_task):
        if send_task._can_retry:
            self._pending_sends.append(send_task)
            if self._resend_timer is None:
                sched = self._scheduler
                self._resend_timer = sched.defer(self._resend_pending, self._delay)
        else:
            send_task._on_error('Send retries exhausted')

    def _resend_pending(self):
        self._resend_timer = None
        self._send_pending()

    def _send_pending(self):
        if self._can_send:
            while self._pending_sends:
                self._send(self._pending_sends.popleft())

    def _open_link(self):
        name = 'openstack.org/om/sender/[%s]/%s' % (self._address, uuid.uuid4().hex)
        link = self._connection.create_sender(name=name, source_address=self._address, target_address=self._address, event_handler=self)
        link.open()
        return link

    def _reopen_link(self):
        if self._connection:
            if self._link:
                self._link.destroy()
            self._link = self._open_link()