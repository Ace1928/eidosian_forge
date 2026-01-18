import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class BasicKeyCommands(CommandsProtocol):
    """
    Redis basic key-based commands
    """

    def append(self, key: KeyT, value: EncodableT) -> ResponseT:
        """
        Appends the string ``value`` to the value at ``key``. If ``key``
        doesn't already exist, create it with a value of ``value``.
        Returns the new length of the value at ``key``.

        For more information see https://redis.io/commands/append
        """
        return self.execute_command('APPEND', key, value)

    def bitcount(self, key: KeyT, start: Union[int, None]=None, end: Union[int, None]=None, mode: Optional[str]=None) -> ResponseT:
        """
        Returns the count of set bits in the value of ``key``.  Optional
        ``start`` and ``end`` parameters indicate which bytes to consider

        For more information see https://redis.io/commands/bitcount
        """
        params = [key]
        if start is not None and end is not None:
            params.append(start)
            params.append(end)
        elif start is not None and end is None or (end is not None and start is None):
            raise DataError('Both start and end must be specified')
        if mode is not None:
            params.append(mode)
        return self.execute_command('BITCOUNT', *params)

    def bitfield(self: Union['Redis', 'AsyncRedis'], key: KeyT, default_overflow: Union[str, None]=None) -> BitFieldOperation:
        """
        Return a BitFieldOperation instance to conveniently construct one or
        more bitfield operations on ``key``.

        For more information see https://redis.io/commands/bitfield
        """
        return BitFieldOperation(self, key, default_overflow=default_overflow)

    def bitfield_ro(self: Union['Redis', 'AsyncRedis'], key: KeyT, encoding: str, offset: BitfieldOffsetT, items: Optional[list]=None) -> ResponseT:
        """
        Return an array of the specified bitfield values
        where the first value is found using ``encoding`` and ``offset``
        parameters and remaining values are result of corresponding
        encoding/offset pairs in optional list ``items``
        Read-only variant of the BITFIELD command.

        For more information see https://redis.io/commands/bitfield_ro
        """
        params = [key, 'GET', encoding, offset]
        items = items or []
        for encoding, offset in items:
            params.extend(['GET', encoding, offset])
        return self.execute_command('BITFIELD_RO', *params)

    def bitop(self, operation: str, dest: KeyT, *keys: KeyT) -> ResponseT:
        """
        Perform a bitwise operation using ``operation`` between ``keys`` and
        store the result in ``dest``.

        For more information see https://redis.io/commands/bitop
        """
        return self.execute_command('BITOP', operation, dest, *keys)

    def bitpos(self, key: KeyT, bit: int, start: Union[int, None]=None, end: Union[int, None]=None, mode: Optional[str]=None) -> ResponseT:
        """
        Return the position of the first bit set to 1 or 0 in a string.
        ``start`` and ``end`` defines search range. The range is interpreted
        as a range of bytes and not a range of bits, so start=0 and end=2
        means to look at the first three bytes.

        For more information see https://redis.io/commands/bitpos
        """
        if bit not in (0, 1):
            raise DataError('bit must be 0 or 1')
        params = [key, bit]
        start is not None and params.append(start)
        if start is not None and end is not None:
            params.append(end)
        elif start is None and end is not None:
            raise DataError('start argument is not set, when end is specified')
        if mode is not None:
            params.append(mode)
        return self.execute_command('BITPOS', *params)

    def copy(self, source: str, destination: str, destination_db: Union[str, None]=None, replace: bool=False) -> ResponseT:
        """
        Copy the value stored in the ``source`` key to the ``destination`` key.

        ``destination_db`` an alternative destination database. By default,
        the ``destination`` key is created in the source Redis database.

        ``replace`` whether the ``destination`` key should be removed before
        copying the value to it. By default, the value is not copied if
        the ``destination`` key already exists.

        For more information see https://redis.io/commands/copy
        """
        params = [source, destination]
        if destination_db is not None:
            params.extend(['DB', destination_db])
        if replace:
            params.append('REPLACE')
        return self.execute_command('COPY', *params)

    def decrby(self, name: KeyT, amount: int=1) -> ResponseT:
        """
        Decrements the value of ``key`` by ``amount``.  If no key exists,
        the value will be initialized as 0 - ``amount``

        For more information see https://redis.io/commands/decrby
        """
        return self.execute_command('DECRBY', name, amount)
    decr = decrby

    def delete(self, *names: KeyT) -> ResponseT:
        """
        Delete one or more keys specified by ``names``
        """
        return self.execute_command('DEL', *names)

    def __delitem__(self, name: KeyT):
        self.delete(name)

    def dump(self, name: KeyT) -> ResponseT:
        """
        Return a serialized version of the value stored at the specified key.
        If key does not exist a nil bulk reply is returned.

        For more information see https://redis.io/commands/dump
        """
        from redis.client import NEVER_DECODE
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command('DUMP', name, **options)

    def exists(self, *names: KeyT) -> ResponseT:
        """
        Returns the number of ``names`` that exist

        For more information see https://redis.io/commands/exists
        """
        return self.execute_command('EXISTS', *names)
    __contains__ = exists

    def expire(self, name: KeyT, time: ExpiryT, nx: bool=False, xx: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
        """
        Set an expire flag on key ``name`` for ``time`` seconds with given
        ``option``. ``time`` can be represented by an integer or a Python timedelta
        object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/expire
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds())
        exp_option = list()
        if nx:
            exp_option.append('NX')
        if xx:
            exp_option.append('XX')
        if gt:
            exp_option.append('GT')
        if lt:
            exp_option.append('LT')
        return self.execute_command('EXPIRE', name, time, *exp_option)

    def expireat(self, name: KeyT, when: AbsExpiryT, nx: bool=False, xx: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
        """
        Set an expire flag on key ``name`` with given ``option``. ``when``
        can be represented as an integer indicating unix time or a Python
        datetime object.

        Valid options are:
            -> NX -- Set expiry only when the key has no expiry
            -> XX -- Set expiry only when the key has an existing expiry
            -> GT -- Set expiry only when the new expiry is greater than current one
            -> LT -- Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/expireat
        """
        if isinstance(when, datetime.datetime):
            when = int(when.timestamp())
        exp_option = list()
        if nx:
            exp_option.append('NX')
        if xx:
            exp_option.append('XX')
        if gt:
            exp_option.append('GT')
        if lt:
            exp_option.append('LT')
        return self.execute_command('EXPIREAT', name, when, *exp_option)

    def expiretime(self, key: str) -> int:
        """
        Returns the absolute Unix timestamp (since January 1, 1970) in seconds
        at which the given key will expire.

        For more information see https://redis.io/commands/expiretime
        """
        return self.execute_command('EXPIRETIME', key)

    def get(self, name: KeyT) -> ResponseT:
        """
        Return the value at key ``name``, or None if the key doesn't exist

        For more information see https://redis.io/commands/get
        """
        return self.execute_command('GET', name)

    def getdel(self, name: KeyT) -> ResponseT:
        """
        Get the value at key ``name`` and delete the key. This command
        is similar to GET, except for the fact that it also deletes
        the key on success (if and only if the key's value type
        is a string).

        For more information see https://redis.io/commands/getdel
        """
        return self.execute_command('GETDEL', name)

    def getex(self, name: KeyT, ex: Union[ExpiryT, None]=None, px: Union[ExpiryT, None]=None, exat: Union[AbsExpiryT, None]=None, pxat: Union[AbsExpiryT, None]=None, persist: bool=False) -> ResponseT:
        """
        Get the value of key and optionally set its expiration.
        GETEX is similar to GET, but is a write command with
        additional options. All time parameters can be given as
        datetime.timedelta or integers.

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
        specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
        specified in unix time.

        ``persist`` remove the time to live associated with ``name``.

        For more information see https://redis.io/commands/getex
        """
        opset = {ex, px, exat, pxat}
        if len(opset) > 2 or (len(opset) > 1 and persist):
            raise DataError('``ex``, ``px``, ``exat``, ``pxat``, and ``persist`` are mutually exclusive.')
        pieces: list[EncodableT] = []
        if ex is not None:
            pieces.append('EX')
            if isinstance(ex, datetime.timedelta):
                ex = int(ex.total_seconds())
            pieces.append(ex)
        if px is not None:
            pieces.append('PX')
            if isinstance(px, datetime.timedelta):
                px = int(px.total_seconds() * 1000)
            pieces.append(px)
        if exat is not None:
            pieces.append('EXAT')
            if isinstance(exat, datetime.datetime):
                exat = int(exat.timestamp())
            pieces.append(exat)
        if pxat is not None:
            pieces.append('PXAT')
            if isinstance(pxat, datetime.datetime):
                pxat = int(pxat.timestamp() * 1000)
            pieces.append(pxat)
        if persist:
            pieces.append('PERSIST')
        return self.execute_command('GETEX', name, *pieces)

    def __getitem__(self, name: KeyT):
        """
        Return the value at key ``name``, raises a KeyError if the key
        doesn't exist.
        """
        value = self.get(name)
        if value is not None:
            return value
        raise KeyError(name)

    def getbit(self, name: KeyT, offset: int) -> ResponseT:
        """
        Returns an integer indicating the value of ``offset`` in ``name``

        For more information see https://redis.io/commands/getbit
        """
        return self.execute_command('GETBIT', name, offset)

    def getrange(self, key: KeyT, start: int, end: int) -> ResponseT:
        """
        Returns the substring of the string value stored at ``key``,
        determined by the offsets ``start`` and ``end`` (both are inclusive)

        For more information see https://redis.io/commands/getrange
        """
        return self.execute_command('GETRANGE', key, start, end)

    def getset(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Sets the value at key ``name`` to ``value``
        and returns the old value at key ``name`` atomically.

        As per Redis 6.2, GETSET is considered deprecated.
        Please use SET with GET parameter in new code.

        For more information see https://redis.io/commands/getset
        """
        return self.execute_command('GETSET', name, value)

    def incrby(self, name: KeyT, amount: int=1) -> ResponseT:
        """
        Increments the value of ``key`` by ``amount``.  If no key exists,
        the value will be initialized as ``amount``

        For more information see https://redis.io/commands/incrby
        """
        return self.execute_command('INCRBY', name, amount)
    incr = incrby

    def incrbyfloat(self, name: KeyT, amount: float=1.0) -> ResponseT:
        """
        Increments the value at key ``name`` by floating ``amount``.
        If no key exists, the value will be initialized as ``amount``

        For more information see https://redis.io/commands/incrbyfloat
        """
        return self.execute_command('INCRBYFLOAT', name, amount)

    def keys(self, pattern: PatternT='*', **kwargs) -> ResponseT:
        """
        Returns a list of keys matching ``pattern``

        For more information see https://redis.io/commands/keys
        """
        return self.execute_command('KEYS', pattern, **kwargs)

    def lmove(self, first_list: str, second_list: str, src: str='LEFT', dest: str='RIGHT') -> ResponseT:
        """
        Atomically returns and removes the first/last element of a list,
        pushing it as the first/last element on the destination list.
        Returns the element being popped and pushed.

        For more information see https://redis.io/commands/lmove
        """
        params = [first_list, second_list, src, dest]
        return self.execute_command('LMOVE', *params)

    def blmove(self, first_list: str, second_list: str, timeout: int, src: str='LEFT', dest: str='RIGHT') -> ResponseT:
        """
        Blocking version of lmove.

        For more information see https://redis.io/commands/blmove
        """
        params = [first_list, second_list, src, dest, timeout]
        return self.execute_command('BLMOVE', *params)

    def mget(self, keys: KeysT, *args: EncodableT) -> ResponseT:
        """
        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/mget
        """
        from redis.client import EMPTY_RESPONSE
        args = list_or_args(keys, args)
        options = {}
        if not args:
            options[EMPTY_RESPONSE] = []
        return self.execute_command('MGET', *args, **options)

    def mset(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ResponseT:
        """
        Sets key/values based on a mapping. Mapping is a dictionary of
        key/value pairs. Both keys and values should be strings or types that
        can be cast to a string via str().

        For more information see https://redis.io/commands/mset
        """
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command('MSET', *items)

    def msetnx(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ResponseT:
        """
        Sets key/values based on a mapping if none of the keys are already set.
        Mapping is a dictionary of key/value pairs. Both keys and values
        should be strings or types that can be cast to a string via str().
        Returns a boolean indicating if the operation was successful.

        For more information see https://redis.io/commands/msetnx
        """
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command('MSETNX', *items)

    def move(self, name: KeyT, db: int) -> ResponseT:
        """
        Moves the key ``name`` to a different Redis database ``db``

        For more information see https://redis.io/commands/move
        """
        return self.execute_command('MOVE', name, db)

    def persist(self, name: KeyT) -> ResponseT:
        """
        Removes an expiration on ``name``

        For more information see https://redis.io/commands/persist
        """
        return self.execute_command('PERSIST', name)

    def pexpire(self, name: KeyT, time: ExpiryT, nx: bool=False, xx: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
        """
        Set an expire flag on key ``name`` for ``time`` milliseconds
        with given ``option``. ``time`` can be represented by an
        integer or a Python timedelta object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/pexpire
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds() * 1000)
        exp_option = list()
        if nx:
            exp_option.append('NX')
        if xx:
            exp_option.append('XX')
        if gt:
            exp_option.append('GT')
        if lt:
            exp_option.append('LT')
        return self.execute_command('PEXPIRE', name, time, *exp_option)

    def pexpireat(self, name: KeyT, when: AbsExpiryT, nx: bool=False, xx: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
        """
        Set an expire flag on key ``name`` with given ``option``. ``when``
        can be represented as an integer representing unix time in
        milliseconds (unix time * 1000) or a Python datetime object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/pexpireat
        """
        if isinstance(when, datetime.datetime):
            when = int(when.timestamp() * 1000)
        exp_option = list()
        if nx:
            exp_option.append('NX')
        if xx:
            exp_option.append('XX')
        if gt:
            exp_option.append('GT')
        if lt:
            exp_option.append('LT')
        return self.execute_command('PEXPIREAT', name, when, *exp_option)

    def pexpiretime(self, key: str) -> int:
        """
        Returns the absolute Unix timestamp (since January 1, 1970) in milliseconds
        at which the given key will expire.

        For more information see https://redis.io/commands/pexpiretime
        """
        return self.execute_command('PEXPIRETIME', key)

    def psetex(self, name: KeyT, time_ms: ExpiryT, value: EncodableT):
        """
        Set the value of key ``name`` to ``value`` that expires in ``time_ms``
        milliseconds. ``time_ms`` can be represented by an integer or a Python
        timedelta object

        For more information see https://redis.io/commands/psetex
        """
        if isinstance(time_ms, datetime.timedelta):
            time_ms = int(time_ms.total_seconds() * 1000)
        return self.execute_command('PSETEX', name, time_ms, value)

    def pttl(self, name: KeyT) -> ResponseT:
        """
        Returns the number of milliseconds until the key ``name`` will expire

        For more information see https://redis.io/commands/pttl
        """
        return self.execute_command('PTTL', name)

    def hrandfield(self, key: str, count: int=None, withvalues: bool=False) -> ResponseT:
        """
        Return a random field from the hash value stored at key.

        count: if the argument is positive, return an array of distinct fields.
        If called with a negative count, the behavior changes and the command
        is allowed to return the same field multiple times. In this case,
        the number of returned fields is the absolute value of the
        specified count.
        withvalues: The optional WITHVALUES modifier changes the reply so it
        includes the respective values of the randomly selected hash fields.

        For more information see https://redis.io/commands/hrandfield
        """
        params = []
        if count is not None:
            params.append(count)
        if withvalues:
            params.append('WITHVALUES')
        return self.execute_command('HRANDFIELD', key, *params)

    def randomkey(self, **kwargs) -> ResponseT:
        """
        Returns the name of a random key

        For more information see https://redis.io/commands/randomkey
        """
        return self.execute_command('RANDOMKEY', **kwargs)

    def rename(self, src: KeyT, dst: KeyT) -> ResponseT:
        """
        Rename key ``src`` to ``dst``

        For more information see https://redis.io/commands/rename
        """
        return self.execute_command('RENAME', src, dst)

    def renamenx(self, src: KeyT, dst: KeyT):
        """
        Rename key ``src`` to ``dst`` if ``dst`` doesn't already exist

        For more information see https://redis.io/commands/renamenx
        """
        return self.execute_command('RENAMENX', src, dst)

    def restore(self, name: KeyT, ttl: float, value: EncodableT, replace: bool=False, absttl: bool=False, idletime: Union[int, None]=None, frequency: Union[int, None]=None) -> ResponseT:
        """
        Create a key using the provided serialized value, previously obtained
        using DUMP.

        ``replace`` allows an existing key on ``name`` to be overridden. If
        it's not specified an error is raised on collision.

        ``absttl`` if True, specified ``ttl`` should represent an absolute Unix
        timestamp in milliseconds in which the key will expire. (Redis 5.0 or
        greater).

        ``idletime`` Used for eviction, this is the number of seconds the
        key must be idle, prior to execution.

        ``frequency`` Used for eviction, this is the frequency counter of
        the object stored at the key, prior to execution.

        For more information see https://redis.io/commands/restore
        """
        params = [name, ttl, value]
        if replace:
            params.append('REPLACE')
        if absttl:
            params.append('ABSTTL')
        if idletime is not None:
            params.append('IDLETIME')
            try:
                params.append(int(idletime))
            except ValueError:
                raise DataError('idletimemust be an integer')
        if frequency is not None:
            params.append('FREQ')
            try:
                params.append(int(frequency))
            except ValueError:
                raise DataError('frequency must be an integer')
        return self.execute_command('RESTORE', *params)

    def set(self, name: KeyT, value: EncodableT, ex: Union[ExpiryT, None]=None, px: Union[ExpiryT, None]=None, nx: bool=False, xx: bool=False, keepttl: bool=False, get: bool=False, exat: Union[AbsExpiryT, None]=None, pxat: Union[AbsExpiryT, None]=None) -> ResponseT:
        """
        Set the value at key ``name`` to ``value``

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``nx`` if set to True, set the value at key ``name`` to ``value`` only
            if it does not exist.

        ``xx`` if set to True, set the value at key ``name`` to ``value`` only
            if it already exists.

        ``keepttl`` if True, retain the time to live associated with the key.
            (Available since Redis 6.0)

        ``get`` if True, set the value at key ``name`` to ``value`` and return
            the old value stored at key, or None if the key did not exist.
            (Available since Redis 6.2)

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
            specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
            specified in unix time.

        For more information see https://redis.io/commands/set
        """
        pieces: list[EncodableT] = [name, value]
        options = {}
        if ex is not None:
            pieces.append('EX')
            if isinstance(ex, datetime.timedelta):
                pieces.append(int(ex.total_seconds()))
            elif isinstance(ex, int):
                pieces.append(ex)
            elif isinstance(ex, str) and ex.isdigit():
                pieces.append(int(ex))
            else:
                raise DataError('ex must be datetime.timedelta or int')
        if px is not None:
            pieces.append('PX')
            if isinstance(px, datetime.timedelta):
                pieces.append(int(px.total_seconds() * 1000))
            elif isinstance(px, int):
                pieces.append(px)
            else:
                raise DataError('px must be datetime.timedelta or int')
        if exat is not None:
            pieces.append('EXAT')
            if isinstance(exat, datetime.datetime):
                exat = int(exat.timestamp())
            pieces.append(exat)
        if pxat is not None:
            pieces.append('PXAT')
            if isinstance(pxat, datetime.datetime):
                pxat = int(pxat.timestamp() * 1000)
            pieces.append(pxat)
        if keepttl:
            pieces.append('KEEPTTL')
        if nx:
            pieces.append('NX')
        if xx:
            pieces.append('XX')
        if get:
            pieces.append('GET')
            options['get'] = True
        return self.execute_command('SET', *pieces, **options)

    def __setitem__(self, name: KeyT, value: EncodableT):
        self.set(name, value)

    def setbit(self, name: KeyT, offset: int, value: int) -> ResponseT:
        """
        Flag the ``offset`` in ``name`` as ``value``. Returns an integer
        indicating the previous value of ``offset``.

        For more information see https://redis.io/commands/setbit
        """
        value = value and 1 or 0
        return self.execute_command('SETBIT', name, offset, value)

    def setex(self, name: KeyT, time: ExpiryT, value: EncodableT) -> ResponseT:
        """
        Set the value of key ``name`` to ``value`` that expires in ``time``
        seconds. ``time`` can be represented by an integer or a Python
        timedelta object.

        For more information see https://redis.io/commands/setex
        """
        if isinstance(time, datetime.timedelta):
            time = int(time.total_seconds())
        return self.execute_command('SETEX', name, time, value)

    def setnx(self, name: KeyT, value: EncodableT) -> ResponseT:
        """
        Set the value of key ``name`` to ``value`` if key doesn't exist

        For more information see https://redis.io/commands/setnx
        """
        return self.execute_command('SETNX', name, value)

    def setrange(self, name: KeyT, offset: int, value: EncodableT) -> ResponseT:
        """
        Overwrite bytes in the value of ``name`` starting at ``offset`` with
        ``value``. If ``offset`` plus the length of ``value`` exceeds the
        length of the original value, the new value will be larger than before.
        If ``offset`` exceeds the length of the original value, null bytes
        will be used to pad between the end of the previous value and the start
        of what's being injected.

        Returns the length of the new string.

        For more information see https://redis.io/commands/setrange
        """
        return self.execute_command('SETRANGE', name, offset, value)

    def stralgo(self, algo: Literal['LCS'], value1: KeyT, value2: KeyT, specific_argument: Union[Literal['strings'], Literal['keys']]='strings', len: bool=False, idx: bool=False, minmatchlen: Union[int, None]=None, withmatchlen: bool=False, **kwargs) -> ResponseT:
        """
        Implements complex algorithms that operate on strings.
        Right now the only algorithm implemented is the LCS algorithm
        (longest common substring). However new algorithms could be
        implemented in the future.

        ``algo`` Right now must be LCS
        ``value1`` and ``value2`` Can be two strings or two keys
        ``specific_argument`` Specifying if the arguments to the algorithm
        will be keys or strings. strings is the default.
        ``len`` Returns just the len of the match.
        ``idx`` Returns the match positions in each string.
        ``minmatchlen`` Restrict the list of matches to the ones of a given
        minimal length. Can be provided only when ``idx`` set to True.
        ``withmatchlen`` Returns the matches with the len of the match.
        Can be provided only when ``idx`` set to True.

        For more information see https://redis.io/commands/stralgo
        """
        supported_algo = ['LCS']
        if algo not in supported_algo:
            supported_algos_str = ', '.join(supported_algo)
            raise DataError(f'The supported algorithms are: {supported_algos_str}')
        if specific_argument not in ['keys', 'strings']:
            raise DataError('specific_argument can be only keys or strings')
        if len and idx:
            raise DataError('len and idx cannot be provided together.')
        pieces: list[EncodableT] = [algo, specific_argument.upper(), value1, value2]
        if len:
            pieces.append(b'LEN')
        if idx:
            pieces.append(b'IDX')
        try:
            int(minmatchlen)
            pieces.extend([b'MINMATCHLEN', minmatchlen])
        except TypeError:
            pass
        if withmatchlen:
            pieces.append(b'WITHMATCHLEN')
        return self.execute_command('STRALGO', *pieces, len=len, idx=idx, minmatchlen=minmatchlen, withmatchlen=withmatchlen, **kwargs)

    def strlen(self, name: KeyT) -> ResponseT:
        """
        Return the number of bytes stored in the value of ``name``

        For more information see https://redis.io/commands/strlen
        """
        return self.execute_command('STRLEN', name)

    def substr(self, name: KeyT, start: int, end: int=-1) -> ResponseT:
        """
        Return a substring of the string at key ``name``. ``start`` and ``end``
        are 0-based integers specifying the portion of the string to return.
        """
        return self.execute_command('SUBSTR', name, start, end)

    def touch(self, *args: KeyT) -> ResponseT:
        """
        Alters the last access time of a key(s) ``*args``. A key is ignored
        if it does not exist.

        For more information see https://redis.io/commands/touch
        """
        return self.execute_command('TOUCH', *args)

    def ttl(self, name: KeyT) -> ResponseT:
        """
        Returns the number of seconds until the key ``name`` will expire

        For more information see https://redis.io/commands/ttl
        """
        return self.execute_command('TTL', name)

    def type(self, name: KeyT) -> ResponseT:
        """
        Returns the type of key ``name``

        For more information see https://redis.io/commands/type
        """
        return self.execute_command('TYPE', name)

    def watch(self, *names: KeyT) -> None:
        """
        Watches the values at keys ``names``, or None if the key doesn't exist

        For more information see https://redis.io/commands/watch
        """
        warnings.warn(DeprecationWarning('Call WATCH from a Pipeline object'))

    def unwatch(self) -> None:
        """
        Unwatches the value at key ``name``, or None of the key doesn't exist

        For more information see https://redis.io/commands/unwatch
        """
        warnings.warn(DeprecationWarning('Call UNWATCH from a Pipeline object'))

    def unlink(self, *names: KeyT) -> ResponseT:
        """
        Unlink one or more keys specified by ``names``

        For more information see https://redis.io/commands/unlink
        """
        return self.execute_command('UNLINK', *names)

    def lcs(self, key1: str, key2: str, len: Optional[bool]=False, idx: Optional[bool]=False, minmatchlen: Optional[int]=0, withmatchlen: Optional[bool]=False) -> Union[str, int, list]:
        """
        Find the longest common subsequence between ``key1`` and ``key2``.
        If ``len`` is true the length of the match will will be returned.
        If ``idx`` is true the match position in each strings will be returned.
        ``minmatchlen`` restrict the list of matches to the ones of
        the given ``minmatchlen``.
        If ``withmatchlen`` the length of the match also will be returned.
        For more information see https://redis.io/commands/lcs
        """
        pieces = [key1, key2]
        if len:
            pieces.append('LEN')
        if idx:
            pieces.append('IDX')
        if minmatchlen != 0:
            pieces.extend(['MINMATCHLEN', minmatchlen])
        if withmatchlen:
            pieces.append('WITHMATCHLEN')
        return self.execute_command('LCS', *pieces)