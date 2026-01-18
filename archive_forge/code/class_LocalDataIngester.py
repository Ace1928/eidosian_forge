import os
import re
import threading
import time
from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat import tf
from tensorboard.data import ingester
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curve_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tb_logging
class LocalDataIngester(ingester.DataIngester):
    """Data ingestion implementation to use when running locally."""

    def __init__(self, flags):
        """Initializes a `LocalDataIngester` from `flags`.

        Args:
          flags: An argparse.Namespace containing TensorBoard CLI flags.

        Returns:
          The new `LocalDataIngester`.
        """
        tensor_size_guidance = dict(DEFAULT_TENSOR_SIZE_GUIDANCE)
        tensor_size_guidance.update(flags.samples_per_plugin)
        self._multiplexer = plugin_event_multiplexer.EventMultiplexer(size_guidance=DEFAULT_SIZE_GUIDANCE, tensor_size_guidance=tensor_size_guidance, purge_orphaned_data=flags.purge_orphaned_data, max_reload_threads=flags.max_reload_threads, event_file_active_filter=_get_event_file_active_filter(flags), detect_file_replacement=flags.detect_file_replacement)
        self._data_provider = data_provider.MultiplexerDataProvider(self._multiplexer, flags.logdir or flags.logdir_spec)
        self._reload_interval = flags.reload_interval
        self._reload_task = flags.reload_task
        if flags.logdir:
            self._path_to_run = {os.path.expanduser(flags.logdir): None}
        else:
            self._path_to_run = _parse_event_files_spec(flags.logdir_spec)
        if getattr(tf, '__version__', 'stub') != 'stub':
            _check_filesystem_support(self._path_to_run.keys())

    @property
    def data_provider(self):
        return self._data_provider

    @property
    def deprecated_multiplexer(self):
        return self._multiplexer

    def start(self):
        """Starts ingesting data based on the ingester flag configuration."""

        def _reload():
            while True:
                start = time.time()
                logger.info('TensorBoard reload process beginning')
                for path, name in self._path_to_run.items():
                    self._multiplexer.AddRunsFromDirectory(path, name)
                logger.info('TensorBoard reload process: Reload the whole Multiplexer')
                self._multiplexer.Reload()
                duration = time.time() - start
                logger.info('TensorBoard done reloading. Load took %0.3f secs', duration)
                if self._reload_interval == 0:
                    break
                time.sleep(self._reload_interval)
        if self._reload_task == 'process':
            logger.info('Launching reload in a child process')
            import multiprocessing
            process = multiprocessing.Process(target=_reload, name='Reloader')
            process.daemon = True
            process.start()
        elif self._reload_task in ('thread', 'auto'):
            logger.info('Launching reload in a daemon thread')
            thread = threading.Thread(target=_reload, name='Reloader')
            thread.daemon = True
            thread.start()
        elif self._reload_task == 'blocking':
            if self._reload_interval != 0:
                raise ValueError('blocking reload only allowed with load_interval=0')
            _reload()
        else:
            raise ValueError('unrecognized reload_task: %s' % self._reload_task)