import asyncio
import copy
import enum
import inspect
import socket
import ssl
import sys
import warnings
import weakref
from abc import abstractmethod
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.compat import Protocol, TypedDict
from redis.connection import DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
from redis.typing import EncodableT
from redis.utils import HIREDIS_AVAILABLE, get_lib_version, str_if_bytes
from .._parsers import (
class AbstractConnection:
    """Manages communication to and from a Redis server"""
    __slots__ = ('db', 'username', 'client_name', 'lib_name', 'lib_version', 'credential_provider', 'password', 'socket_timeout', 'socket_connect_timeout', 'redis_connect_func', 'retry_on_timeout', 'retry_on_error', 'health_check_interval', 'next_health_check', 'last_active_at', 'encoder', 'ssl_context', 'protocol', '_reader', '_writer', '_parser', '_connect_callbacks', '_buffer_cutoff', '_lock', '_socket_read_size', '__dict__')

    def __init__(self, *, db: Union[str, int]=0, password: Optional[str]=None, socket_timeout: Optional[float]=None, socket_connect_timeout: Optional[float]=None, retry_on_timeout: bool=False, retry_on_error: Union[list, _Sentinel]=SENTINEL, encoding: str='utf-8', encoding_errors: str='strict', decode_responses: bool=False, parser_class: Type[BaseParser]=DefaultParser, socket_read_size: int=65536, health_check_interval: float=0, client_name: Optional[str]=None, lib_name: Optional[str]='redis-py', lib_version: Optional[str]=get_lib_version(), username: Optional[str]=None, retry: Optional[Retry]=None, redis_connect_func: Optional[ConnectCallbackT]=None, encoder_class: Type[Encoder]=Encoder, credential_provider: Optional[CredentialProvider]=None, protocol: Optional[int]=2):
        if (username or password) and credential_provider is not None:
            raise DataError("'username' and 'password' cannot be passed along with 'credential_provider'. Please provide only one of the following arguments: \n1. 'password' and (optional) 'username'\n2. 'credential_provider'")
        self.db = db
        self.client_name = client_name
        self.lib_name = lib_name
        self.lib_version = lib_version
        self.credential_provider = credential_provider
        self.password = password
        self.username = username
        self.socket_timeout = socket_timeout
        if socket_connect_timeout is None:
            socket_connect_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        if retry_on_error is SENTINEL:
            retry_on_error = []
        if retry_on_timeout:
            retry_on_error.append(TimeoutError)
            retry_on_error.append(socket.timeout)
            retry_on_error.append(asyncio.TimeoutError)
        self.retry_on_error = retry_on_error
        if retry or retry_on_error:
            if not retry:
                self.retry = Retry(NoBackoff(), 1)
            else:
                self.retry = copy.deepcopy(retry)
            self.retry.update_supported_errors(retry_on_error)
        else:
            self.retry = Retry(NoBackoff(), 0)
        self.health_check_interval = health_check_interval
        self.next_health_check: float = -1
        self.encoder = encoder_class(encoding, encoding_errors, decode_responses)
        self.redis_connect_func = redis_connect_func
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._socket_read_size = socket_read_size
        self.set_parser(parser_class)
        self._connect_callbacks: List[weakref.WeakMethod[ConnectCallbackT]] = []
        self._buffer_cutoff = 6000
        try:
            p = int(protocol)
        except TypeError:
            p = DEFAULT_RESP_VERSION
        except ValueError:
            raise ConnectionError('protocol must be an integer')
        finally:
            if p < 2 or p > 3:
                raise ConnectionError('protocol must be either 2 or 3')
            self.protocol = protocol

    def __del__(self, _warnings: Any=warnings):
        if getattr(self, '_writer', None):
            _warnings.warn(f'unclosed Connection {self!r}', ResourceWarning, source=self)
            self._close()

    def _close(self):
        """
        Internal method to silently close the connection without waiting
        """
        if self._writer:
            self._writer.close()
            self._writer = self._reader = None

    def __repr__(self):
        repr_args = ','.join((f'{k}={v}' for k, v in self.repr_pieces()))
        return f'{self.__class__.__name__}<{repr_args}>'

    @abstractmethod
    def repr_pieces(self):
        pass

    @property
    def is_connected(self):
        return self._reader is not None and self._writer is not None

    def register_connect_callback(self, callback):
        """
        Register a callback to be called when the connection is established either
        initially or reconnected.  This allows listeners to issue commands that
        are ephemeral to the connection, for example pub/sub subscription or
        key tracking.  The callback must be a _method_ and will be kept as
        a weak reference.
        """
        wm = weakref.WeakMethod(callback)
        if wm not in self._connect_callbacks:
            self._connect_callbacks.append(wm)

    def deregister_connect_callback(self, callback):
        """
        De-register a previously registered callback.  It will no-longer receive
        notifications on connection events.  Calling this is not required when the
        listener goes away, since the callbacks are kept as weak methods.
        """
        try:
            self._connect_callbacks.remove(weakref.WeakMethod(callback))
        except ValueError:
            pass

    def set_parser(self, parser_class: Type[BaseParser]) -> None:
        """
        Creates a new instance of parser_class with socket size:
        _socket_read_size and assigns it to the parser for the connection
        :param parser_class: The required parser class
        """
        self._parser = parser_class(socket_read_size=self._socket_read_size)

    async def connect(self):
        """Connects to the Redis server if not already connected"""
        if self.is_connected:
            return
        try:
            await self.retry.call_with_retry(lambda: self._connect(), lambda error: self.disconnect())
        except asyncio.CancelledError:
            raise
        except (socket.timeout, asyncio.TimeoutError):
            raise TimeoutError('Timeout connecting to server')
        except OSError as e:
            raise ConnectionError(self._error_message(e))
        except Exception as exc:
            raise ConnectionError(exc) from exc
        try:
            if not self.redis_connect_func:
                await self.on_connect()
            else:
                await self.redis_connect_func(self) if asyncio.iscoroutinefunction(self.redis_connect_func) else self.redis_connect_func(self)
        except RedisError:
            await self.disconnect()
            raise
        self._connect_callbacks = [ref for ref in self._connect_callbacks if ref()]
        for ref in self._connect_callbacks:
            callback = ref()
            task = callback(self)
            if task and inspect.isawaitable(task):
                await task

    @abstractmethod
    async def _connect(self):
        pass

    @abstractmethod
    def _host_error(self) -> str:
        pass

    @abstractmethod
    def _error_message(self, exception: BaseException) -> str:
        pass

    async def on_connect(self) -> None:
        """Initialize the connection, authenticate and select a database"""
        self._parser.on_connect(self)
        parser = self._parser
        auth_args = None
        if self.credential_provider or (self.username or self.password):
            cred_provider = self.credential_provider or UsernamePasswordCredentialProvider(self.username, self.password)
            auth_args = cred_provider.get_credentials()
        if auth_args and self.protocol not in [2, '2']:
            if isinstance(self._parser, _AsyncRESP2Parser):
                self.set_parser(_AsyncRESP3Parser)
                self._parser.EXCEPTION_CLASSES = parser.EXCEPTION_CLASSES
                self._parser.on_connect(self)
            if len(auth_args) == 1:
                auth_args = ['default', auth_args[0]]
            await self.send_command('HELLO', self.protocol, 'AUTH', *auth_args)
            response = await self.read_response()
            if response.get(b'proto') != int(self.protocol) and response.get('proto') != int(self.protocol):
                raise ConnectionError('Invalid RESP version')
        elif auth_args:
            await self.send_command('AUTH', *auth_args, check_health=False)
            try:
                auth_response = await self.read_response()
            except AuthenticationWrongNumberOfArgsError:
                await self.send_command('AUTH', auth_args[-1], check_health=False)
                auth_response = await self.read_response()
            if str_if_bytes(auth_response) != 'OK':
                raise AuthenticationError('Invalid Username or Password')
        elif self.protocol not in [2, '2']:
            if isinstance(self._parser, _AsyncRESP2Parser):
                self.set_parser(_AsyncRESP3Parser)
                self._parser.EXCEPTION_CLASSES = parser.EXCEPTION_CLASSES
                self._parser.on_connect(self)
            await self.send_command('HELLO', self.protocol)
            response = await self.read_response()
        if self.client_name:
            await self.send_command('CLIENT', 'SETNAME', self.client_name)
            if str_if_bytes(await self.read_response()) != 'OK':
                raise ConnectionError('Error setting client name')
        if self.lib_name:
            await self.send_command('CLIENT', 'SETINFO', 'LIB-NAME', self.lib_name)
        if self.lib_version:
            await self.send_command('CLIENT', 'SETINFO', 'LIB-VER', self.lib_version)
        if self.db:
            await self.send_command('SELECT', self.db)
        for _ in (sent for sent in (self.lib_name, self.lib_version) if sent):
            try:
                await self.read_response()
            except ResponseError:
                pass
        if self.db:
            if str_if_bytes(await self.read_response()) != 'OK':
                raise ConnectionError('Invalid Database')

    async def disconnect(self, nowait: bool=False) -> None:
        """Disconnects from the Redis server"""
        try:
            async with async_timeout(self.socket_connect_timeout):
                self._parser.on_disconnect()
                if not self.is_connected:
                    return
                try:
                    self._writer.close()
                    if not nowait:
                        await self._writer.wait_closed()
                except OSError:
                    pass
                finally:
                    self._reader = None
                    self._writer = None
        except asyncio.TimeoutError:
            raise TimeoutError(f'Timed out closing connection after {self.socket_connect_timeout}') from None

    async def _send_ping(self):
        """Send PING, expect PONG in return"""
        await self.send_command('PING', check_health=False)
        if str_if_bytes(await self.read_response()) != 'PONG':
            raise ConnectionError('Bad response from PING health check')

    async def _ping_failed(self, error):
        """Function to call when PING fails"""
        await self.disconnect()

    async def check_health(self):
        """Check the health of the connection with a PING/PONG"""
        if self.health_check_interval and asyncio.get_running_loop().time() > self.next_health_check:
            await self.retry.call_with_retry(self._send_ping, self._ping_failed)

    async def _send_packed_command(self, command: Iterable[bytes]) -> None:
        self._writer.writelines(command)
        await self._writer.drain()

    async def send_packed_command(self, command: Union[bytes, str, Iterable[bytes]], check_health: bool=True) -> None:
        if not self.is_connected:
            await self.connect()
        elif check_health:
            await self.check_health()
        try:
            if isinstance(command, str):
                command = command.encode()
            if isinstance(command, bytes):
                command = [command]
            if self.socket_timeout:
                await asyncio.wait_for(self._send_packed_command(command), self.socket_timeout)
            else:
                self._writer.writelines(command)
                await self._writer.drain()
        except asyncio.TimeoutError:
            await self.disconnect(nowait=True)
            raise TimeoutError('Timeout writing to socket') from None
        except OSError as e:
            await self.disconnect(nowait=True)
            if len(e.args) == 1:
                err_no, errmsg = ('UNKNOWN', e.args[0])
            else:
                err_no = e.args[0]
                errmsg = e.args[1]
            raise ConnectionError(f'Error {err_no} while writing to socket. {errmsg}.') from e
        except BaseException:
            await self.disconnect(nowait=True)
            raise

    async def send_command(self, *args: Any, **kwargs: Any) -> None:
        """Pack and send a command to the Redis server"""
        await self.send_packed_command(self.pack_command(*args), check_health=kwargs.get('check_health', True))

    async def can_read_destructive(self):
        """Poll the socket to see if there's data that can be read."""
        try:
            return await self._parser.can_read_destructive()
        except OSError as e:
            await self.disconnect(nowait=True)
            host_error = self._host_error()
            raise ConnectionError(f'Error while reading from {host_error}: {e.args}')

    async def read_response(self, disable_decoding: bool=False, timeout: Optional[float]=None, *, disconnect_on_error: bool=True, push_request: Optional[bool]=False):
        """Read the response from a previously sent command"""
        read_timeout = timeout if timeout is not None else self.socket_timeout
        host_error = self._host_error()
        try:
            if read_timeout is not None and self.protocol in ['3', 3] and (not HIREDIS_AVAILABLE):
                async with async_timeout(read_timeout):
                    response = await self._parser.read_response(disable_decoding=disable_decoding, push_request=push_request)
            elif read_timeout is not None:
                async with async_timeout(read_timeout):
                    response = await self._parser.read_response(disable_decoding=disable_decoding)
            elif self.protocol in ['3', 3] and (not HIREDIS_AVAILABLE):
                response = await self._parser.read_response(disable_decoding=disable_decoding, push_request=push_request)
            else:
                response = await self._parser.read_response(disable_decoding=disable_decoding)
        except asyncio.TimeoutError:
            if timeout is not None:
                return None
            if disconnect_on_error:
                await self.disconnect(nowait=True)
            raise TimeoutError(f'Timeout reading from {host_error}')
        except OSError as e:
            if disconnect_on_error:
                await self.disconnect(nowait=True)
            raise ConnectionError(f'Error while reading from {host_error} : {e.args}')
        except BaseException:
            if disconnect_on_error:
                await self.disconnect(nowait=True)
            raise
        if self.health_check_interval:
            next_time = asyncio.get_running_loop().time() + self.health_check_interval
            self.next_health_check = next_time
        if isinstance(response, ResponseError):
            raise response from None
        return response

    def pack_command(self, *args: EncodableT) -> List[bytes]:
        """Pack a series of arguments into the Redis protocol"""
        output = []
        assert not isinstance(args[0], float)
        if isinstance(args[0], str):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b' ' in args[0]:
            args = tuple(args[0].split()) + args[1:]
        buff = SYM_EMPTY.join((SYM_STAR, str(len(args)).encode(), SYM_CRLF))
        buffer_cutoff = self._buffer_cutoff
        for arg in map(self.encoder.encode, args):
            arg_length = len(arg)
            if len(buff) > buffer_cutoff or arg_length > buffer_cutoff or isinstance(arg, memoryview):
                buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF))
                output.append(buff)
                output.append(arg)
                buff = SYM_CRLF
            else:
                buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF, arg, SYM_CRLF))
        output.append(buff)
        return output

    def pack_commands(self, commands: Iterable[Iterable[EncodableT]]) -> List[bytes]:
        """Pack multiple commands into the Redis protocol"""
        output: List[bytes] = []
        pieces: List[bytes] = []
        buffer_length = 0
        buffer_cutoff = self._buffer_cutoff
        for cmd in commands:
            for chunk in self.pack_command(*cmd):
                chunklen = len(chunk)
                if buffer_length > buffer_cutoff or chunklen > buffer_cutoff or isinstance(chunk, memoryview):
                    if pieces:
                        output.append(SYM_EMPTY.join(pieces))
                    buffer_length = 0
                    pieces = []
                if chunklen > buffer_cutoff or isinstance(chunk, memoryview):
                    output.append(chunk)
                else:
                    pieces.append(chunk)
                    buffer_length += chunklen
        if pieces:
            output.append(SYM_EMPTY.join(pieces))
        return output