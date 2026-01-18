import collections
import functools
import imghdr
import mimetypes
import os
import threading
import numpy as np
from werkzeug import wrappers
from google.protobuf import json_format
from google.protobuf import text_format
from tensorboard import context
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins.projector import metadata
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.util import tb_logging
class ProjectorPlugin(base_plugin.TBPlugin):
    """Embedding projector."""
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates ProjectorPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self.data_provider = context.data_provider
        self.logdir = context.logdir
        self.readers = {}
        self._run_paths = None
        self._configs = {}
        self.config_fpaths = None
        self.tensor_cache = LRUCache(_TENSOR_CACHE_CAPACITY)
        self._is_active = False
        self._thread_for_determining_is_active = None

    def get_plugin_apps(self):
        asset_prefix = 'tf_projector_plugin'
        return {RUNS_ROUTE: self._serve_runs, CONFIG_ROUTE: self._serve_config, TENSOR_ROUTE: self._serve_tensor, METADATA_ROUTE: self._serve_metadata, BOOKMARKS_ROUTE: self._serve_bookmarks, SPRITE_IMAGE_ROUTE: self._serve_sprite_image, '/index.js': functools.partial(self._serve_file, os.path.join(asset_prefix, 'index.js')), '/projector_binary.html': functools.partial(self._serve_file, os.path.join(asset_prefix, 'projector_binary.html')), '/projector_binary.js': functools.partial(self._serve_file, os.path.join(asset_prefix, 'projector_binary.js'))}

    def is_active(self):
        """Determines whether this plugin is active.

        This plugin is only active if any run has an embedding, and only
        when running against a local log directory.

        Returns:
          Whether any run has embedding data to show in the projector.
        """
        if not self.data_provider or not self.logdir:
            return False
        if self._is_active:
            return True
        if self._thread_for_determining_is_active:
            return self._is_active
        new_thread = threading.Thread(target=self._determine_is_active, name='ProjectorPluginIsActiveThread')
        self._thread_for_determining_is_active = new_thread
        new_thread.start()
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path='/index.js', disable_reload=True)

    def _determine_is_active(self):
        """Determines whether the plugin is active.

        This method is run in a separate thread so that the plugin can
        offer an immediate response to whether it is active and
        determine whether it should be active in a separate thread.
        """
        self._update_configs()
        if self._configs:
            self._is_active = True
        self._thread_for_determining_is_active = None

    def _update_configs(self):
        """Updates `self._configs` and `self._run_paths`."""
        if self.data_provider and self.logdir:
            ctx = context.RequestContext()
            run_paths = {run.run_name: os.path.join(self.logdir, run.run_name) for run in self.data_provider.list_runs(ctx, experiment_id='')}
        else:
            run_paths = {}
        run_paths_changed = run_paths != self._run_paths
        self._run_paths = run_paths
        run_path_pairs = list(self._run_paths.items())
        self._append_plugin_asset_directories(run_path_pairs)
        if '.' not in self._run_paths:
            run_path_pairs.append(('.', self.logdir))
        if run_paths_changed or _latest_checkpoints_changed(self._configs, run_path_pairs):
            self.readers = {}
            self._configs, self.config_fpaths = self._read_latest_config_files(run_path_pairs)
            self._augment_configs_with_checkpoint_info()

    def _augment_configs_with_checkpoint_info(self):
        for run, config in self._configs.items():
            for embedding in config.embeddings:
                if embedding.tensor_name.endswith(':0'):
                    embedding.tensor_name = embedding.tensor_name[:-2]
                if embedding.tensor_path:
                    fpath = _rel_to_abs_asset_path(embedding.tensor_path, self.config_fpaths[run])
                    tensor = self.tensor_cache.get((run, embedding.tensor_name))
                    if tensor is None:
                        try:
                            tensor = _read_tensor_tsv_file(fpath)
                        except UnicodeDecodeError:
                            tensor = _read_tensor_binary_file(fpath, embedding.tensor_shape)
                        self.tensor_cache.set((run, embedding.tensor_name), tensor)
                    if not embedding.tensor_shape:
                        embedding.tensor_shape.extend([len(tensor), len(tensor[0])])
            reader = self._get_reader_for_run(run)
            if not reader:
                continue
            special_embedding = None
            if config.embeddings and (not config.embeddings[0].tensor_name):
                special_embedding = config.embeddings[0]
                config.embeddings.remove(special_embedding)
            var_map = reader.get_variable_to_shape_map()
            for tensor_name, tensor_shape in var_map.items():
                if len(tensor_shape) != 2:
                    continue
                if '.OPTIMIZER_SLOT' in tensor_name:
                    continue
                embedding = self._get_embedding(tensor_name, config)
                if not embedding:
                    embedding = config.embeddings.add()
                    embedding.tensor_name = tensor_name
                    if special_embedding:
                        embedding.metadata_path = special_embedding.metadata_path
                        embedding.bookmarks_path = special_embedding.bookmarks_path
                if not embedding.tensor_shape:
                    embedding.tensor_shape.extend(tensor_shape)
        runs_to_remove = []
        for run, config in self._configs.items():
            if not config.embeddings:
                runs_to_remove.append(run)
        for run in runs_to_remove:
            del self._configs[run]
            del self.config_fpaths[run]

    def _read_latest_config_files(self, run_path_pairs):
        """Reads and returns the projector config files in every run
        directory."""
        configs = {}
        config_fpaths = {}
        for run_name, assets_dir in run_path_pairs:
            config = ProjectorConfig()
            config_fpath = os.path.join(assets_dir, metadata.PROJECTOR_FILENAME)
            if tf.io.gfile.exists(config_fpath):
                with tf.io.gfile.GFile(config_fpath, 'r') as f:
                    file_content = f.read()
                text_format.Parse(file_content, config)
            has_tensor_files = False
            for embedding in config.embeddings:
                if embedding.tensor_path:
                    if not embedding.tensor_name:
                        embedding.tensor_name = os.path.basename(embedding.tensor_path)
                    has_tensor_files = True
                    break
            if not config.model_checkpoint_path:
                logdir = _assets_dir_to_logdir(assets_dir)
                ckpt_path = _find_latest_checkpoint(logdir)
                if not ckpt_path and (not has_tensor_files):
                    continue
                if ckpt_path:
                    config.model_checkpoint_path = ckpt_path
            if config.model_checkpoint_path and _using_tf() and (not tf.io.gfile.glob(config.model_checkpoint_path + '*')):
                logger.warning('Checkpoint file "%s" not found', config.model_checkpoint_path)
                continue
            configs[run_name] = config
            config_fpaths[run_name] = config_fpath
        return (configs, config_fpaths)

    def _get_reader_for_run(self, run):
        if run in self.readers:
            return self.readers[run]
        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path and _using_tf():
            try:
                reader = tf.train.load_checkpoint(config.model_checkpoint_path)
            except Exception:
                logger.warning('Failed reading "%s"', config.model_checkpoint_path)
        self.readers[run] = reader
        return reader

    def _get_metadata_file_for_tensor(self, tensor_name, config):
        embedding_info = self._get_embedding(tensor_name, config)
        if embedding_info:
            return embedding_info.metadata_path
        return None

    def _get_bookmarks_file_for_tensor(self, tensor_name, config):
        embedding_info = self._get_embedding(tensor_name, config)
        if embedding_info:
            return embedding_info.bookmarks_path
        return None

    def _canonical_tensor_name(self, tensor_name):
        if ':' not in tensor_name:
            return tensor_name + ':0'
        else:
            return tensor_name

    def _get_embedding(self, tensor_name, config):
        if not config.embeddings:
            return None
        for info in config.embeddings:
            if self._canonical_tensor_name(info.tensor_name) == self._canonical_tensor_name(tensor_name):
                return info
        return None

    def _append_plugin_asset_directories(self, run_path_pairs):
        extra = []
        plugin_assets_name = metadata.PLUGIN_ASSETS_NAME
        for run, logdir in run_path_pairs:
            assets = plugin_asset_util.ListAssets(logdir, plugin_assets_name)
            if metadata.PROJECTOR_FILENAME not in assets:
                continue
            assets_dir = os.path.join(self._run_paths[run], metadata.PLUGINS_DIR, plugin_assets_name)
            assets_path_pair = (run, os.path.abspath(assets_dir))
            extra.append(assets_path_pair)
        run_path_pairs.extend(extra)

    @wrappers.Request.application
    def _serve_file(self, file_path, request):
        """Returns a resource file."""
        res_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(res_path, 'rb') as read_file:
            mimetype = mimetypes.guess_type(file_path)[0]
            return Respond(request, read_file.read(), content_type=mimetype)

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Returns a list of runs that have embeddings."""
        self._update_configs()
        return Respond(request, list(self._configs.keys()), 'application/json')

    @wrappers.Request.application
    def _serve_config(self, request):
        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        self._update_configs()
        config = self._configs.get(run)
        if config is None:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
        return Respond(request, json_format.MessageToJson(config), 'application/json')

    @wrappers.Request.application
    def _serve_metadata(self, request):
        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
        num_rows = _parse_positive_int_param(request, 'num_rows')
        if num_rows == -1:
            return Respond(request, 'query parameter num_rows must be integer > 0', 'text/plain', 400)
        self._update_configs()
        config = self._configs.get(run)
        if config is None:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
        fpath = self._get_metadata_file_for_tensor(name, config)
        if not fpath:
            return Respond(request, 'No metadata file found for tensor "%s" in the config file "%s"' % (name, self.config_fpaths[run]), 'text/plain', 400)
        fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
        if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
            return Respond(request, '"%s" not found, or is not a file' % fpath, 'text/plain', 400)
        num_header_rows = 0
        with tf.io.gfile.GFile(fpath, 'r') as f:
            lines = []
            for line in f:
                lines.append(line)
                if len(lines) == 1 and '\t' in lines[0]:
                    num_header_rows = 1
                if num_rows and len(lines) >= num_rows + num_header_rows:
                    break
        return Respond(request, ''.join(lines), 'text/plain')

    @wrappers.Request.application
    def _serve_tensor(self, request):
        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
        num_rows = _parse_positive_int_param(request, 'num_rows')
        if num_rows == -1:
            return Respond(request, 'query parameter num_rows must be integer > 0', 'text/plain', 400)
        self._update_configs()
        config = self._configs.get(run)
        if config is None:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
        tensor = self.tensor_cache.get((run, name))
        if tensor is None:
            embedding = self._get_embedding(name, config)
            if embedding and embedding.tensor_path:
                fpath = _rel_to_abs_asset_path(embedding.tensor_path, self.config_fpaths[run])
                if not tf.io.gfile.exists(fpath):
                    return Respond(request, 'Tensor file "%s" does not exist' % fpath, 'text/plain', 400)
                try:
                    tensor = _read_tensor_tsv_file(fpath)
                except UnicodeDecodeError:
                    tensor = _read_tensor_binary_file(fpath, embedding.tensor_shape)
            else:
                reader = self._get_reader_for_run(run)
                if not reader or not reader.has_tensor(name):
                    return Respond(request, 'Tensor "%s" not found in checkpoint dir "%s"' % (name, config.model_checkpoint_path), 'text/plain', 400)
                try:
                    tensor = reader.get_tensor(name)
                except tf.errors.InvalidArgumentError as e:
                    return Respond(request, str(e), 'text/plain', 400)
            self.tensor_cache.set((run, name), tensor)
        if num_rows:
            tensor = tensor[:num_rows]
        if tensor.dtype != 'float32':
            tensor = tensor.astype(dtype='float32', copy=False)
        data_bytes = tensor.tobytes()
        return Respond(request, data_bytes, 'application/octet-stream')

    @wrappers.Request.application
    def _serve_bookmarks(self, request):
        run = request.args.get('run')
        if not run:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
        self._update_configs()
        config = self._configs.get(run)
        if config is None:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
        fpath = self._get_bookmarks_file_for_tensor(name, config)
        if not fpath:
            return Respond(request, 'No bookmarks file found for tensor "%s" in the config file "%s"' % (name, self.config_fpaths[run]), 'text/plain', 400)
        fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
        if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
            return Respond(request, '"%s" not found, or is not a file' % fpath, 'text/plain', 400)
        bookmarks_json = None
        with tf.io.gfile.GFile(fpath, 'rb') as f:
            bookmarks_json = f.read()
        return Respond(request, bookmarks_json, 'application/json')

    @wrappers.Request.application
    def _serve_sprite_image(self, request):
        run = request.args.get('run')
        if not run:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
        self._update_configs()
        config = self._configs.get(run)
        if config is None:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
        embedding_info = self._get_embedding(name, config)
        if not embedding_info or not embedding_info.sprite.image_path:
            return Respond(request, 'No sprite image file found for tensor "%s" in the config file "%s"' % (name, self.config_fpaths[run]), 'text/plain', 400)
        fpath = os.path.expanduser(embedding_info.sprite.image_path)
        fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
        if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
            return Respond(request, '"%s" does not exist or is directory' % fpath, 'text/plain', 400)
        f = tf.io.gfile.GFile(fpath, 'rb')
        encoded_image_string = f.read()
        f.close()
        image_type = imghdr.what(None, encoded_image_string)
        mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
        return Respond(request, encoded_image_string, mime_type)