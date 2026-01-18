from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
@final
class DTLSChannel(trio.abc.Channel[bytes], metaclass=NoPublicConstructor):
    """A DTLS connection.

    This class has no public constructor – you get instances by calling
    `DTLSEndpoint.serve` or `~DTLSEndpoint.connect`.

    .. attribute:: endpoint

       The `DTLSEndpoint` that this connection is using.

    .. attribute:: peer_address

       The IP/port of the remote peer that this connection is associated with.

    """

    def __init__(self, endpoint: DTLSEndpoint, peer_address: Any, ctx: SSL.Context) -> None:
        self.endpoint = endpoint
        self.peer_address = peer_address
        self._packets_dropped_in_trio = 0
        self._client_hello = None
        self._did_handshake = False
        ctx.set_options(SSL.OP_NO_QUERY_MTU | SSL.OP_NO_RENEGOTIATION)
        self._ssl = SSL.Connection(ctx)
        self._handshake_mtu = 0
        self.set_ciphertext_mtu(best_guess_mtu(self.endpoint.socket))
        self._replaced = False
        self._closed = False
        self._q = _Queue[bytes](endpoint.incoming_packets_buffer)
        self._handshake_lock = trio.Lock()
        self._record_encoder: RecordEncoder = RecordEncoder()
        self._final_volley: list[_AnyHandshakeMessage] = []

    def _set_replaced(self) -> None:
        self._replaced = True
        self._q.s.close()

    def _check_replaced(self) -> None:
        if self._replaced:
            raise trio.BrokenResourceError('peer tore down this connection to start a new one')

    def close(self) -> None:
        """Close this connection.

        `DTLSChannel`\\s don't actually own any OS-level resources – the
        socket is owned by the `DTLSEndpoint`, not the individual connections. So
        you don't really *have* to call this. But it will interrupt any other tasks
        calling `receive` with a `ClosedResourceError`, and cause future attempts to use
        this connection to fail.

        You can also use this object as a synchronous or asynchronous context manager.

        """
        if self._closed:
            return
        self._closed = True
        if self.endpoint._streams.get(self.peer_address) is self:
            del self.endpoint._streams[self.peer_address]
        self._q.r.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        return self.close()

    async def aclose(self) -> None:
        """Close this connection, but asynchronously.

        This is included to satisfy the `trio.abc.Channel` contract. It's
        identical to `close`, but async.

        """
        self.close()
        await trio.lowlevel.checkpoint()

    async def _send_volley(self, volley_messages: list[_AnyHandshakeMessage]) -> None:
        packets = self._record_encoder.encode_volley(volley_messages, self._handshake_mtu)
        for packet in packets:
            async with self.endpoint._send_lock:
                await self.endpoint.socket.sendto(packet, self.peer_address)

    async def _resend_final_volley(self) -> None:
        await self._send_volley(self._final_volley)

    async def do_handshake(self, *, initial_retransmit_timeout: float=1.0) -> None:
        """Perform the handshake.

        Calling this is optional – if you don't, then it will be automatically called
        the first time you call `send` or `receive`. But calling it explicitly can be
        useful in case you want to control the retransmit timeout, use a cancel scope to
        place an overall timeout on the handshake, or catch errors from the handshake
        specifically.

        It's safe to call this multiple times, or call it simultaneously from multiple
        tasks – the first call will perform the handshake, and the rest will be no-ops.

        Args:

          initial_retransmit_timeout (float): Since UDP is an unreliable protocol, it's
            possible that some of the packets we send during the handshake will get
            lost. To handle this, DTLS uses a timer to automatically retransmit
            handshake packets that don't receive a response. This lets you set the
            timeout we use to detect packet loss. Ideally, it should be set to ~1.5
            times the round-trip time to your peer, but 1 second is a reasonable
            default. There's `some useful guidance here
            <https://tlswg.org/dtls13-spec/draft-ietf-tls-dtls13.html#name-timer-values>`__.

            This is the *initial* timeout, because if packets keep being lost then Trio
            will automatically back off to longer values, to avoid overloading the
            network.

        """
        async with self._handshake_lock:
            if self._did_handshake:
                return
            timeout = initial_retransmit_timeout
            volley_messages: list[_AnyHandshakeMessage] = []
            volley_failed_sends = 0

            def read_volley() -> list[_AnyHandshakeMessage]:
                volley_bytes = _read_loop(self._ssl.bio_read)
                new_volley_messages = decode_volley_trusted(volley_bytes)
                if new_volley_messages and volley_messages and isinstance(new_volley_messages[0], HandshakeMessage) and isinstance(volley_messages[0], HandshakeMessage) and (new_volley_messages[0].msg_seq == volley_messages[0].msg_seq):
                    return []
                else:
                    return new_volley_messages
            with contextlib.suppress(SSL.WantReadError):
                self._ssl.do_handshake()
            volley_messages = read_volley()
            if not volley_messages:
                raise SSL.Error("something wrong with peer's ClientHello")
            while True:
                assert volley_messages
                self._check_replaced()
                await self._send_volley(volley_messages)
                self.endpoint._ensure_receive_loop()
                with trio.move_on_after(timeout) as cscope:
                    async for packet in self._q.r:
                        self._ssl.bio_write(packet)
                        try:
                            self._ssl.do_handshake()
                        except (SSL.WantReadError, SSL.Error):
                            pass
                        else:
                            self._did_handshake = True
                            self._final_volley = read_volley()
                            await self._send_volley(self._final_volley)
                            return
                        maybe_volley = read_volley()
                        if maybe_volley:
                            if isinstance(maybe_volley[0], PseudoHandshakeMessage) and maybe_volley[0].content_type == ContentType.alert:
                                await self._send_volley(maybe_volley)
                            else:
                                volley_messages = maybe_volley
                                if volley_failed_sends == 0:
                                    timeout = initial_retransmit_timeout
                                volley_failed_sends = 0
                                break
                    else:
                        assert self._replaced
                        self._check_replaced()
                if cscope.cancelled_caught:
                    timeout = min(2 * timeout, 60.0)
                    volley_failed_sends += 1
                    if volley_failed_sends == 2:
                        self._handshake_mtu = min(self._handshake_mtu, worst_case_mtu(self.endpoint.socket))

    async def send(self, data: bytes) -> None:
        """Send a packet of data, securely."""
        if self._closed:
            raise trio.ClosedResourceError
        if not data:
            raise ValueError("openssl doesn't support sending empty DTLS packets")
        if not self._did_handshake:
            await self.do_handshake()
        self._check_replaced()
        self._ssl.write(data)
        async with self.endpoint._send_lock:
            await self.endpoint.socket.sendto(_read_loop(self._ssl.bio_read), self.peer_address)

    async def receive(self) -> bytes:
        """Fetch the next packet of data from this connection's peer, waiting if
        necessary.

        This is safe to call from multiple tasks simultaneously, in case you have some
        reason to do that. And more importantly, it's cancellation-safe, meaning that
        cancelling a call to `receive` will never cause a packet to be lost or corrupt
        the underlying connection.

        """
        if not self._did_handshake:
            await self.do_handshake()
        while True:
            try:
                packet = await self._q.r.receive()
            except trio.EndOfChannel:
                assert self._replaced
                self._check_replaced()
            self._ssl.bio_write(packet)
            cleartext = _read_loop(self._ssl.read)
            if cleartext:
                return cleartext

    def set_ciphertext_mtu(self, new_mtu: int) -> None:
        """Tells Trio the `largest amount of data that can be sent in a single packet to
        this peer <https://en.wikipedia.org/wiki/Maximum_transmission_unit>`__.

        Trio doesn't actually enforce this limit – if you pass a huge packet to `send`,
        then we'll dutifully encrypt it and attempt to send it. But calling this method
        does have two useful effects:

        - If called before the handshake is performed, then Trio will automatically
          fragment handshake messages to fit within the given MTU. It also might
          fragment them even smaller, if it detects signs of packet loss, so setting
          this should never be necessary to make a successful connection. But, the
          packet loss detection only happens after multiple timeouts have expired, so if
          you have reason to believe that a smaller MTU is required, then you can set
          this to skip those timeouts and establish the connection more quickly.

        - It changes the value returned from `get_cleartext_mtu`. So if you have some
          kind of estimate of the network-level MTU, then you can use this to figure out
          how much overhead DTLS will need for hashes/padding/etc., and how much space
          you have left for your application data.

        The MTU here is measuring the largest UDP *payload* you think can be sent, the
        amount of encrypted data that can be handed to the operating system in a single
        call to `send`. It should *not* include IP/UDP headers. Note that OS estimates
        of the MTU often are link-layer MTUs, so you have to subtract off 28 bytes on
        IPv4 and 48 bytes on IPv6 to get the ciphertext MTU.

        By default, Trio assumes an MTU of 1472 bytes on IPv4, and 1452 bytes on IPv6,
        which correspond to the common Ethernet MTU of 1500 bytes after accounting for
        IP/UDP overhead.

        """
        self._handshake_mtu = new_mtu
        self._ssl.set_ciphertext_mtu(new_mtu)

    def get_cleartext_mtu(self) -> int:
        """Returns the largest number of bytes that you can pass in a single call to
        `send` while still fitting within the network-level MTU.

        See `set_ciphertext_mtu` for more details.

        """
        if not self._did_handshake:
            raise trio.NeedHandshakeError
        return self._ssl.get_cleartext_mtu()

    def statistics(self) -> DTLSChannelStatistics:
        """Returns a `DTLSChannelStatistics` object with statistics about this connection."""
        return DTLSChannelStatistics(self._packets_dropped_in_trio)