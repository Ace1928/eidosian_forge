import errno
import logging
import os
import subprocess
import tempfile
import time
import grpc
import pkg_resources
from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging
class SubprocessServerDataIngester(ingester.DataIngester):
    """Start a new data server as a subprocess."""

    def __init__(self, server_binary, logdir, *, reload_interval, channel_creds_type, samples_per_plugin=None, extra_flags=None):
        """Initializes an ingester with the given configuration.

        Args:
          server_binary: `ServerBinary` to launch.
          logdir: String, as passed to `--logdir`.
          reload_interval: Number, as passed to `--reload_interval`.
          channel_creds_type: `grpc_util.ChannelCredsType`, as passed to
            `--grpc_creds_type`.
          samples_per_plugin: Dict[String, Int], as parsed from
            `--samples_per_plugin`.
          extra_flags: List of extra string flags to be passed to the
            data server without further interpretation.
        """
        self._server_binary = server_binary
        self._data_provider = None
        self._logdir = logdir
        self._reload_interval = reload_interval
        self._channel_creds_type = channel_creds_type
        self._samples_per_plugin = samples_per_plugin or {}
        self._extra_flags = list(extra_flags or [])

    @property
    def data_provider(self):
        if self._data_provider is None:
            raise RuntimeError('Must call `start` first')
        return self._data_provider

    def start(self):
        if self._data_provider:
            return
        tmpdir = tempfile.TemporaryDirectory(prefix='tensorboard_data_server_')
        port_file_path = os.path.join(tmpdir.name, 'port')
        error_file_path = os.path.join(tmpdir.name, 'startup_error')
        if self._reload_interval <= 0:
            reload = 'once'
        else:
            reload = str(int(self._reload_interval))
        sample_hint_pairs = ['%s=%s' % (k, 'all' if v == 0 else v) for k, v in self._samples_per_plugin.items()]
        samples_per_plugin = ','.join(sample_hint_pairs)
        args = [self._server_binary.path, '--logdir=%s' % os.path.expanduser(self._logdir), '--reload=%s' % reload, '--samples-per-plugin=%s' % samples_per_plugin, '--port=0', '--port-file=%s' % (port_file_path,), '--die-after-stdin']
        if self._server_binary.at_least_version('0.5.0a0'):
            args.append('--error-file=%s' % (error_file_path,))
        if logger.isEnabledFor(logging.INFO):
            args.append('--verbose')
        if logger.isEnabledFor(logging.DEBUG):
            args.append('--verbose')
        args.extend(self._extra_flags)
        logger.info('Spawning data server: %r', args)
        popen = subprocess.Popen(args, stdin=subprocess.PIPE)
        self._stdin_handle = popen.stdin
        port = None
        time.sleep(0.01)
        for i in range(20):
            if popen.poll() is not None:
                msg = (_maybe_read_file(error_file_path) or '').strip()
                if not msg:
                    msg = 'exited with %d; check stderr for details' % popen.poll()
                raise DataServerStartupError(msg)
            logger.info('Polling for data server port (attempt %d)', i)
            port_file_contents = _maybe_read_file(port_file_path)
            logger.info('Port file contents: %r', port_file_contents)
            if (port_file_contents or '').endswith('\n'):
                port = int(port_file_contents)
                break
            time.sleep(0.5)
        if port is None:
            raise DataServerStartupError('Timed out while waiting for data server to start. It may still be running as pid %d.' % popen.pid)
        addr = 'localhost:%d' % port
        stub = _make_stub(addr, self._channel_creds_type)
        logger.info('Opened channel to data server at pid %d via %s', popen.pid, addr)
        req = data_provider_pb2.GetExperimentRequest()
        try:
            stub.GetExperiment(req, timeout=5)
        except grpc.RpcError as e:
            msg = 'Failed to communicate with data server at %s: %s' % (addr, e)
            logging.warning('%s', msg)
            raise DataServerStartupError(msg) from e
        logger.info('Got valid response from data server')
        self._data_provider = grpc_provider.GrpcDataProvider(addr, stub)