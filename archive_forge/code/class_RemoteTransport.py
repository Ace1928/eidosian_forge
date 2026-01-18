imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class RemoteTransport(transport.ConnectedTransport):
    """Connection to a smart server.

    The connection holds references to the medium that can be used to send
    requests to the server.

    The connection has a notion of the current directory to which it's
    connected; this is incorporated in filenames passed to the server.

    This supports some higher-level RPC operations and can also be treated
    like a Transport to do file-like operations.

    The connection can be made over a tcp socket, an ssh pipe or a series of
    http requests.  There are concrete subclasses for each type:
    RemoteTCPTransport, etc.
    """
    _max_readv_bytes = 5 * 1024 * 1024

    def __init__(self, url, _from_transport=None, medium=None, _client=None):
        """Constructor.

        :param _from_transport: Another RemoteTransport instance that this
            one is being cloned from.  Attributes such as the medium will
            be reused.

        :param medium: The medium to use for this RemoteTransport.  If None,
            the medium from the _from_transport is shared.  If both this
            and _from_transport are None, a new medium will be built.
            _from_transport and medium cannot both be specified.

        :param _client: Override the _SmartClient used by this transport.  This
            should only be used for testing purposes; normally this is
            determined from the medium.
        """
        super().__init__(url, _from_transport=_from_transport)
        if _from_transport is not None and isinstance(_from_transport, RemoteTransport):
            _client = _from_transport._client
        elif _from_transport is None:
            credentials = None
            if medium is None:
                medium, credentials = self._build_medium()
                if 'hpss' in debug.debug_flags:
                    trace.mutter('hpss: Built a new medium: %s', medium.__class__.__name__)
            self._shared_connection = transport._SharedConnection(medium, credentials, self.base)
        elif medium is None:
            medium = self._shared_connection.connection
        else:
            raise AssertionError('Both _from_transport (%r) and medium (%r) passed to RemoteTransport.__init__, but these parameters are mutally exclusive.' % (_from_transport, medium))
        if _client is None:
            self._client = client._SmartClient(medium)
        else:
            self._client = _client

    def _build_medium(self):
        """Create the medium if _from_transport does not provide one.

        The medium is analogous to the connection for ConnectedTransport: it
        allows connection sharing.
        """
        return (None, None)

    def _report_activity(self, bytes, direction):
        """See Transport._report_activity.

        Does nothing; the smart medium will report activity triggered by a
        RemoteTransport.
        """
        pass

    def is_readonly(self):
        """Smart server transport can do read/write file operations."""
        try:
            resp = self._call2(b'Transport.is_readonly')
        except errors.UnknownSmartMethod:
            return False
        if resp == (b'yes',):
            return True
        elif resp == (b'no',):
            return False
        else:
            raise errors.UnexpectedSmartServerResponse(resp)

    def get_smart_client(self):
        return self._get_connection()

    def get_smart_medium(self):
        return self._get_connection()

    def _remote_path(self, relpath):
        """Returns the Unicode version of the absolute path for relpath."""
        path = urlutils.URL._combine_paths(self._parsed_url.path, relpath)
        if not isinstance(path, bytes):
            path = path.encode()
        return path

    def _call(self, method, *args):
        resp = self._call2(method, *args)
        self._ensure_ok(resp)

    def _call2(self, method, *args):
        """Call a method on the remote server."""
        try:
            return self._client.call(method, *args)
        except errors.ErrorFromSmartServer as err:
            if args:
                context = {'relpath': args[0].decode('utf-8')}
            else:
                context = {}
            self._translate_error(err, **context)

    def _call_with_body_bytes(self, method, args, body):
        """Call a method on the remote server with body bytes."""
        try:
            return self._client.call_with_body_bytes(method, args, body)
        except errors.ErrorFromSmartServer as err:
            if args:
                context = {'relpath': args[0]}
            else:
                context = {}
            self._translate_error(err, **context)

    def has(self, relpath):
        """Indicate whether a remote file of the given name exists or not.

        :see: Transport.has()
        """
        resp = self._call2(b'has', self._remote_path(relpath))
        if resp == (b'yes',):
            return True
        elif resp == (b'no',):
            return False
        else:
            raise errors.UnexpectedSmartServerResponse(resp)

    def get(self, relpath):
        """Return file-like object reading the contents of a remote file.

        :see: Transport.get_bytes()/get_file()
        """
        return BytesIO(self.get_bytes(relpath))

    def get_bytes(self, relpath):
        remote = self._remote_path(relpath)
        try:
            resp, response_handler = self._client.call_expecting_body(b'get', remote)
        except errors.ErrorFromSmartServer as err:
            self._translate_error(err, relpath)
        if resp != (b'ok',):
            response_handler.cancel_read_body()
            raise errors.UnexpectedSmartServerResponse(resp)
        return response_handler.read_body_bytes()

    def _serialise_optional_mode(self, mode):
        if mode is None:
            return b''
        else:
            return ('%d' % mode).encode('ascii')

    def mkdir(self, relpath, mode=None):
        resp = self._call2(b'mkdir', self._remote_path(relpath), self._serialise_optional_mode(mode))

    def open_write_stream(self, relpath, mode=None):
        """See Transport.open_write_stream."""
        self.put_bytes(relpath, b'', mode)
        result = transport.AppendBasedFileStream(self, relpath)
        transport._file_streams[self.abspath(relpath)] = result
        return result

    def put_bytes(self, relpath: str, raw_bytes: bytes, mode=None):
        if not isinstance(raw_bytes, bytes):
            raise TypeError('raw_bytes must be bytes string, not %s' % type(raw_bytes))
        resp = self._call_with_body_bytes(b'put', (self._remote_path(relpath), self._serialise_optional_mode(mode)), raw_bytes)
        self._ensure_ok(resp)
        return len(raw_bytes)

    def put_bytes_non_atomic(self, relpath: str, raw_bytes: bytes, mode=None, create_parent_dir=False, dir_mode=None):
        """See Transport.put_bytes_non_atomic."""
        create_parent_str = b'F'
        if create_parent_dir:
            create_parent_str = b'T'
        resp = self._call_with_body_bytes(b'put_non_atomic', (self._remote_path(relpath), self._serialise_optional_mode(mode), create_parent_str, self._serialise_optional_mode(dir_mode)), raw_bytes)
        self._ensure_ok(resp)

    def put_file(self, relpath, upload_file, mode=None):
        pos = upload_file.tell()
        try:
            return self.put_bytes(relpath, upload_file.read(), mode)
        except:
            upload_file.seek(pos)
            raise

    def put_file_non_atomic(self, relpath, f, mode=None, create_parent_dir=False, dir_mode=None):
        return self.put_bytes_non_atomic(relpath, f.read(), mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)

    def append_file(self, relpath, from_file, mode=None):
        return self.append_bytes(relpath, from_file.read(), mode)

    def append_bytes(self, relpath, bytes, mode=None):
        resp = self._call_with_body_bytes(b'append', (self._remote_path(relpath), self._serialise_optional_mode(mode)), bytes)
        if resp[0] == b'appended':
            return int(resp[1])
        raise errors.UnexpectedSmartServerResponse(resp)

    def delete(self, relpath):
        resp = self._call2(b'delete', self._remote_path(relpath))
        self._ensure_ok(resp)

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        return self.base

    def recommended_page_size(self):
        """Return the recommended page size for this transport."""
        return 64 * 1024

    def _readv(self, relpath, offsets):
        if not offsets:
            return
        offsets = list(offsets)
        sorted_offsets = sorted(offsets)
        coalesced = list(self._coalesce_offsets(sorted_offsets, limit=self._max_readv_combine, fudge_factor=self._bytes_to_read_before_seek, max_size=self._max_readv_bytes))
        requests = []
        cur_request = []
        cur_len = 0
        for c in coalesced:
            if c.length + cur_len > self._max_readv_bytes:
                requests.append(cur_request)
                cur_request = [c]
                cur_len = c.length
                continue
            cur_request.append(c)
            cur_len += c.length
        if cur_request:
            requests.append(cur_request)
        if 'hpss' in debug.debug_flags:
            trace.mutter('%s.readv %s offsets => %s coalesced => %s requests (%s)', self.__class__.__name__, len(offsets), len(coalesced), len(requests), sum(map(len, requests)))
        data_map = {}
        offset_stack = iter(offsets)
        next_offset = [next(offset_stack)]
        for cur_request in requests:
            try:
                result = self._client.call_with_body_readv_array((b'readv', self._remote_path(relpath)), [(c.start, c.length) for c in cur_request])
                resp, response_handler = result
            except errors.ErrorFromSmartServer as err:
                self._translate_error(err, relpath)
            if resp[0] != b'readv':
                response_handler.cancel_read_body()
                raise errors.UnexpectedSmartServerResponse(resp)
            yield from self._handle_response(offset_stack, cur_request, response_handler, data_map, next_offset)

    def _handle_response(self, offset_stack, coalesced, response_handler, data_map, next_offset):
        cur_offset_and_size = next_offset[0]
        data = response_handler.read_body_bytes()
        data_offset = 0
        for c_offset in coalesced:
            if len(data) < c_offset.length:
                raise errors.ShortReadvError(relpath, c_offset.start, c_offset.length, actual=len(data))
            for suboffset, subsize in c_offset.ranges:
                key = (c_offset.start + suboffset, subsize)
                this_data = data[data_offset + suboffset:data_offset + suboffset + subsize]
                if key == cur_offset_and_size:
                    yield (cur_offset_and_size[0], this_data)
                    try:
                        cur_offset_and_size = next_offset[0] = next(offset_stack)
                    except StopIteration:
                        return
                else:
                    data_map[key] = this_data
            data_offset += c_offset.length
            while cur_offset_and_size in data_map:
                this_data = data_map.pop(cur_offset_and_size)
                yield (cur_offset_and_size[0], this_data)
                try:
                    cur_offset_and_size = next_offset[0] = next(offset_stack)
                except StopIteration:
                    return

    def rename(self, rel_from, rel_to):
        self._call(b'rename', self._remote_path(rel_from), self._remote_path(rel_to))

    def move(self, rel_from, rel_to):
        self._call(b'move', self._remote_path(rel_from), self._remote_path(rel_to))

    def rmdir(self, relpath):
        resp = self._call(b'rmdir', self._remote_path(relpath))

    def _ensure_ok(self, resp):
        if resp[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(resp)

    def _translate_error(self, err, relpath=None):
        remote._translate_error(err, path=relpath)

    def disconnect(self):
        m = self.get_smart_medium()
        if m is not None:
            m.disconnect()

    def stat(self, relpath):
        resp = self._call2(b'stat', self._remote_path(relpath))
        if resp[0] == b'stat':
            return _SmartStat(int(resp[1]), int(resp[2], 8))
        raise errors.UnexpectedSmartServerResponse(resp)

    def listable(self):
        return True

    def list_dir(self, relpath):
        resp = self._call2(b'list_dir', self._remote_path(relpath))
        if resp[0] == b'names':
            return [name.decode('utf-8') for name in resp[1:]]
        raise errors.UnexpectedSmartServerResponse(resp)

    def iter_files_recursive(self):
        resp = self._call2(b'iter_files_recursive', self._remote_path(''))
        if resp[0] == b'names':
            return [name.decode('utf-8') for name in resp[1:]]
        raise errors.UnexpectedSmartServerResponse(resp)