import copy
import json
import logging
import os
import platform
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import wandb
import wandb.env
from wandb import trigger
from wandb.errors import CommError, Error, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.integration import sagemaker
from wandb.integration.magic import magic_install
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import StrPath
from wandb.util import _is_artifact_representation
from . import wandb_login, wandb_setup
from .backend.backend import Backend
from .lib import (
from .lib.deprecate import Deprecated, deprecate
from .lib.mailbox import Mailbox, MailboxProgress
from .lib.printer import Printer, get_printer
from .lib.wburls import wburls
from .wandb_helper import parse_config
from .wandb_run import Run, TeardownHook, TeardownStage
from .wandb_settings import Settings, Source
class _WandbInit:
    _init_telemetry_obj: telemetry.TelemetryRecord

    def __init__(self) -> None:
        self.kwargs = None
        self.settings: Optional[Settings] = None
        self.sweep_config: Dict[str, Any] = {}
        self.launch_config: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.run: Optional[Run] = None
        self.backend: Optional[Backend] = None
        self._teardown_hooks: List[TeardownHook] = []
        self._wl: Optional[wandb_setup._WandbSetup] = None
        self._reporter: Optional[wandb.sdk.lib.reporting.Reporter] = None
        self.notebook: Optional[wandb.jupyter.Notebook] = None
        self.printer: Optional[Printer] = None
        self._init_telemetry_obj = telemetry.TelemetryRecord()
        self.deprecated_features_used: Dict[str, str] = dict()

    def _setup_printer(self, settings: Settings) -> None:
        if self.printer:
            return
        self.printer = get_printer(settings._jupyter)

    def setup(self, kwargs: Any) -> None:
        """Complete setup for `wandb.init()`.

        This includes parsing all arguments, applying them with settings and enabling logging.
        """
        self.kwargs = kwargs
        singleton = wandb_setup._WandbSetup._instance
        if singleton is not None:
            self._setup_printer(settings=singleton._settings)
            assert self.printer
            exclude_env_vars = {'WANDB_SERVICE', 'WANDB_KUBEFLOW_URL'}
            singleton_env = {k: v for k, v in singleton._environ.items() if k.startswith('WANDB_') and k not in exclude_env_vars}
            os_env = {k: v for k, v in os.environ.items() if k.startswith('WANDB_') and k not in exclude_env_vars}
            if set(singleton_env.keys()) != set(os_env.keys()) or set(singleton_env.values()) != set(os_env.values()):
                line = f'Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to {self.printer.link(wburls.get('wandb_init'), 'the W&B docs')}.'
                self.printer.display(line, level='warn')
        mode = kwargs.get('mode')
        settings_mode = (kwargs.get('settings') or {}).get('mode')
        _disable_service = mode == 'disabled' or settings_mode == 'disabled'
        setup_settings = {'_disable_service': _disable_service}
        self._wl = wandb_setup.setup(settings=setup_settings)
        assert self._wl is not None
        _set_logger(self._wl._get_logger())
        settings: Settings = self._wl.settings.copy()
        if settings.launch and singleton is not None:
            settings.update({'run_id': None}, source=Source.INIT)
        settings_param = kwargs.pop('settings', None)
        if settings_param is not None and isinstance(settings_param, (Settings, dict)):
            settings.update(settings_param, source=Source.INIT)
        self._setup_printer(settings)
        self._reporter = reporting.setup_reporter(settings=settings)
        sagemaker_config: Dict = dict() if settings.sagemaker_disable else sagemaker.parse_sm_config()
        if sagemaker_config:
            sagemaker_api_key = sagemaker_config.get('wandb_api_key', None)
            sagemaker_run, sagemaker_env = sagemaker.parse_sm_resources()
            if sagemaker_env:
                if sagemaker_api_key:
                    sagemaker_env['WANDB_API_KEY'] = sagemaker_api_key
                settings._apply_env_vars(sagemaker_env)
                wandb.setup(settings=settings)
            settings.update(sagemaker_run, source=Source.SETUP)
            with telemetry.context(obj=self._init_telemetry_obj) as tel:
                tel.feature.sagemaker = True
        with telemetry.context(obj=self._init_telemetry_obj) as tel:
            if kwargs.get('config'):
                tel.feature.set_init_config = True
            if kwargs.get('name'):
                tel.feature.set_init_name = True
            if kwargs.get('id'):
                tel.feature.set_init_id = True
            if kwargs.get('tags'):
                tel.feature.set_init_tags = True
        init_config = kwargs.pop('config', None) or dict()
        deprecated_kwargs = {'config_include_keys': "Use `config=wandb.helper.parse_config(config_object, include=('key',))` instead.", 'config_exclude_keys': "Use `config=wandb.helper.parse_config(config_object, exclude=('key',))` instead."}
        for deprecated_kwarg, msg in deprecated_kwargs.items():
            if kwargs.get(deprecated_kwarg):
                self.deprecated_features_used[deprecated_kwarg] = msg
        init_config = parse_config(init_config, include=kwargs.pop('config_include_keys', None), exclude=kwargs.pop('config_exclude_keys', None))
        self.sweep_config = dict()
        sweep_config = self._wl._sweep_config or dict()
        self.config = dict()
        self.init_artifact_config: Dict[str, Any] = dict()
        for config_data in (sagemaker_config, self._wl._config, init_config):
            if not config_data:
                continue
            self._split_artifacts_from_config(config_data, self.config)
        if sweep_config:
            self._split_artifacts_from_config(sweep_config, self.sweep_config)
        monitor_gym = kwargs.pop('monitor_gym', None)
        if monitor_gym and len(wandb.patched['gym']) == 0:
            wandb.gym.monitor()
        if wandb.patched['tensorboard']:
            with telemetry.context(obj=self._init_telemetry_obj) as tel:
                tel.feature.tensorboard_patch = True
        tensorboard = kwargs.pop('tensorboard', None)
        sync_tensorboard = kwargs.pop('sync_tensorboard', None)
        if tensorboard or (sync_tensorboard and len(wandb.patched['tensorboard']) == 0):
            wandb.tensorboard.patch()
            with telemetry.context(obj=self._init_telemetry_obj) as tel:
                tel.feature.tensorboard_sync = True
        magic = kwargs.get('magic')
        if magic not in (None, False):
            magic_install(kwargs)
        init_settings = {key: kwargs[key] for key in ['anonymous', 'force', 'mode', 'resume'] if kwargs.get(key) is not None}
        if init_settings:
            settings.update(init_settings, source=Source.INIT)
        if not settings._offline and (not settings._noop):
            wandb_login._login(anonymous=kwargs.pop('anonymous', None), force=kwargs.pop('force', None), _disable_warning=True, _silent=settings.quiet or settings.silent, _entity=kwargs.get('entity') or settings.entity)
        wl = wandb.setup()
        assert wl is not None
        settings._apply_settings(wl.settings)
        save_code_pre_user_settings = settings.save_code
        settings._apply_init(kwargs)
        if not settings._offline and (not settings._noop):
            user_settings = self._wl._load_user_settings()
            settings._apply_user(user_settings)
        if save_code_pre_user_settings is False:
            settings.update({'save_code': False}, source=Source.INIT)
        settings._set_run_start_time(source=Source.INIT)
        if not settings._noop:
            self._log_setup(settings)
            if settings._jupyter:
                self._jupyter_setup(settings)
        launch_config = _handle_launch_config(settings)
        if launch_config:
            self._split_artifacts_from_config(launch_config, self.launch_config)
        self.settings = settings

    def teardown(self) -> None:
        assert logger
        logger.info('tearing down wandb.init')
        for hook in self._teardown_hooks:
            hook.call()

    def _split_artifacts_from_config(self, config_source: dict, config_target: dict) -> None:
        for k, v in config_source.items():
            if _is_artifact_representation(v):
                self.init_artifact_config[k] = v
            else:
                config_target.setdefault(k, v)

    def _enable_logging(self, log_fname: str, run_id: Optional[str]=None) -> None:
        """Enable logging to the global debug log.

        This adds a run_id to the log, in case of multiple processes on the same machine.
        Currently, there is no way to disable logging after it's enabled.
        """
        handler = logging.FileHandler(log_fname)
        handler.setLevel(logging.INFO)

        class WBFilter(logging.Filter):

            def filter(self, record: logging.LogRecord) -> bool:
                record.run_id = run_id
                return True
        if run_id:
            formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(threadName)-10s:%(process)d [%(run_id)s:%(filename)s:%(funcName)s():%(lineno)s] %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(threadName)-10s:%(process)d [%(filename)s:%(funcName)s():%(lineno)s] %(message)s')
        handler.setFormatter(formatter)
        if run_id:
            handler.addFilter(WBFilter())
        assert logger is not None
        logger.propagate = False
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self._teardown_hooks.append(TeardownHook(lambda: (handler.close(), logger.removeHandler(handler)), TeardownStage.LATE))

    def _safe_symlink(self, base: str, target: str, name: str, delete: bool=False) -> None:
        if not hasattr(os, 'symlink'):
            return
        pid = os.getpid()
        tmp_name = os.path.join(base, '%s.%d' % (name, pid))
        if delete:
            try:
                os.remove(os.path.join(base, name))
            except OSError:
                pass
        target = os.path.relpath(target, base)
        try:
            os.symlink(target, tmp_name)
            os.rename(tmp_name, os.path.join(base, name))
        except OSError:
            pass

    def _pause_backend(self, *args: Any, **kwargs: Any) -> None:
        if self.backend is None:
            return None
        if self.notebook.save_ipynb():
            assert self.run is not None
            res = self.run.log_code(root=None)
            logger.info('saved code: %s', res)
        if self.backend.interface is not None:
            logger.info('pausing backend')
            self.backend.interface.publish_pause()

    def _resume_backend(self, *args: Any, **kwargs: Any) -> None:
        if self.backend is not None and self.backend.interface is not None:
            logger.info('resuming backend')
            self.backend.interface.publish_resume()

    def _jupyter_teardown(self) -> None:
        """Teardown hooks and display saving, called with wandb.finish."""
        assert self.notebook
        ipython = self.notebook.shell
        self.notebook.save_history()
        if self.notebook.save_ipynb():
            assert self.run is not None
            res = self.run.log_code(root=None)
            logger.info('saved code and history: %s', res)
        logger.info('cleaning up jupyter logic')
        for hook in ipython.events.callbacks['pre_run_cell']:
            if '_resume_backend' in hook.__name__:
                ipython.events.unregister('pre_run_cell', hook)
        for hook in ipython.events.callbacks['post_run_cell']:
            if '_pause_backend' in hook.__name__:
                ipython.events.unregister('post_run_cell', hook)
        ipython.display_pub.publish = ipython.display_pub._orig_publish
        del ipython.display_pub._orig_publish

    def _jupyter_setup(self, settings: Settings) -> None:
        """Add hooks, and session history saving."""
        self.notebook = wandb.jupyter.Notebook(settings)
        ipython = self.notebook.shell
        if not hasattr(ipython.display_pub, '_orig_publish'):
            logger.info('configuring jupyter hooks %s', self)
            ipython.display_pub._orig_publish = ipython.display_pub.publish
            ipython.events.register('pre_run_cell', self._resume_backend)
            ipython.events.register('post_run_cell', self._pause_backend)
            self._teardown_hooks.append(TeardownHook(self._jupyter_teardown, TeardownStage.EARLY))

        def publish(data, metadata=None, **kwargs) -> None:
            ipython.display_pub._orig_publish(data, metadata=metadata, **kwargs)
            assert self.notebook is not None
            self.notebook.save_display(ipython.execution_count, {'data': data, 'metadata': metadata})
        ipython.display_pub.publish = publish

    def _log_setup(self, settings: Settings) -> None:
        """Set up logging from settings."""
        filesystem.mkdir_exists_ok(os.path.dirname(settings.log_user))
        filesystem.mkdir_exists_ok(os.path.dirname(settings.log_internal))
        filesystem.mkdir_exists_ok(os.path.dirname(settings.sync_file))
        filesystem.mkdir_exists_ok(settings.files_dir)
        filesystem.mkdir_exists_ok(settings._tmp_code_dir)
        if settings.symlink:
            self._safe_symlink(os.path.dirname(settings.sync_symlink_latest), os.path.dirname(settings.sync_file), os.path.basename(settings.sync_symlink_latest), delete=True)
            self._safe_symlink(os.path.dirname(settings.log_symlink_user), settings.log_user, os.path.basename(settings.log_symlink_user), delete=True)
            self._safe_symlink(os.path.dirname(settings.log_symlink_internal), settings.log_internal, os.path.basename(settings.log_symlink_internal), delete=True)
        _set_logger(logging.getLogger('wandb'))
        self._enable_logging(settings.log_user)
        assert self._wl
        assert logger
        self._wl._early_logger_flush(logger)
        logger.info(f'Logging user logs to {settings.log_user}')
        logger.info(f'Logging internal logs to {settings.log_internal}')

    def _make_run_disabled(self) -> RunDisabled:
        drun = RunDisabled()
        drun.config = wandb.wandb_sdk.wandb_config.Config()
        drun.config.update(self.sweep_config)
        drun.config.update(self.config)
        drun.summary = SummaryDisabled()
        drun.log = lambda data, *_, **__: drun.summary.update(data)
        drun.finish = lambda *_, **__: module.unset_globals()
        drun.step = 0
        drun.resumed = False
        drun.disabled = True
        drun.id = runid.generate_id()
        drun.name = 'dummy-' + drun.id
        drun.dir = tempfile.gettempdir()
        module.set_global(run=drun, config=drun.config, log=drun.log, summary=drun.summary, save=drun.save, use_artifact=drun.use_artifact, log_artifact=drun.log_artifact, define_metric=drun.define_metric, plot_table=drun.plot_table, alert=drun.alert)
        return drun

    def _on_progress_init(self, handle: MailboxProgress) -> None:
        assert self.printer
        line = 'Waiting for wandb.init()...\r'
        percent_done = handle.percent_done
        self.printer.progress_update(line, percent_done=percent_done)

    def init(self) -> Union[Run, RunDisabled, None]:
        if logger is None:
            raise RuntimeError('Logger not initialized')
        logger.info('calling init triggers')
        trigger.call('on_init', **self.kwargs)
        assert self.settings is not None
        assert self._wl is not None
        assert self._reporter is not None
        logger.info(f'wandb.init called with sweep_config: {self.sweep_config}\nconfig: {self.config}')
        if self.settings._noop:
            return self._make_run_disabled()
        if self.settings.reinit or (self.settings._jupyter and self.settings.reinit is not False):
            if len(self._wl._global_run_stack) > 0:
                if len(self._wl._global_run_stack) > 1:
                    wandb.termwarn('If you want to track multiple runs concurrently in wandb, you should use multi-processing not threads')
                latest_run = self._wl._global_run_stack[-1]
                logger.info(f're-initializing run, found existing run on stack: {latest_run._run_id}')
                jupyter = self.settings._jupyter
                if jupyter and (not self.settings.silent):
                    ipython.display_html(f'Finishing last run (ID:{latest_run._run_id}) before initializing another...')
                latest_run.finish()
                if jupyter and (not self.settings.silent):
                    ipython.display_html(f'Successfully finished last run (ID:{latest_run._run_id}). Initializing new run:<br/>')
        elif isinstance(wandb.run, Run):
            manager = self._wl._get_manager()
            if not manager or os.getpid() == wandb.run._init_pid:
                logger.info('wandb.init() called when a run is still active')
                with telemetry.context() as tel:
                    tel.feature.init_return_run = True
                return wandb.run
        logger.info('starting backend')
        manager = self._wl._get_manager()
        if manager:
            logger.info('setting up manager')
            manager._inform_init(settings=self.settings.to_proto(), run_id=self.settings.run_id)
        mailbox = Mailbox()
        backend = Backend(settings=self.settings, manager=manager, mailbox=mailbox)
        backend.ensure_launched()
        logger.info('backend started and connected')
        run = Run(config=self.config, settings=self.settings, sweep_config=self.sweep_config, launch_config=self.launch_config)
        with telemetry.context(run=run, obj=self._init_telemetry_obj) as tel:
            tel.cli_version = wandb.__version__
            tel.python_version = platform.python_version()
            tel.platform = f'{platform.system()}-{platform.machine()}'.lower()
            hf_version = _huggingface_version()
            if hf_version:
                tel.huggingface_version = hf_version
            if self.settings._jupyter:
                tel.env.jupyter = True
            if self.settings._ipython:
                tel.env.ipython = True
            if self.settings._colab:
                tel.env.colab = True
            if self.settings._kaggle:
                tel.env.kaggle = True
            if self.settings._windows:
                tel.env.windows = True
            if self.settings.launch:
                tel.feature.launch = True
            if self.settings._async_upload_concurrency_limit:
                tel.feature.async_uploads = True
            for module_name in telemetry.list_telemetry_imports(only_imported=True):
                setattr(tel.imports_init, module_name, True)
            active_start_method: Optional[str] = None
            if self.settings.start_method == 'thread':
                active_start_method = self.settings.start_method
            else:
                active_start_method = getattr(backend._multiprocessing, 'get_start_method', lambda: None)()
            if active_start_method == 'spawn':
                tel.env.start_spawn = True
            elif active_start_method == 'fork':
                tel.env.start_fork = True
            elif active_start_method == 'forkserver':
                tel.env.start_forkserver = True
            elif active_start_method == 'thread':
                tel.env.start_thread = True
            if os.environ.get('PEX'):
                tel.env.pex = True
            if self.settings._aws_lambda:
                tel.env.aws_lambda = True
            if os.environ.get(wandb.env._DISABLE_SERVICE):
                tel.feature.service_disabled = True
            if manager:
                tel.feature.service = True
            if self.settings._flow_control_disabled:
                tel.feature.flow_control_disabled = True
            if self.settings._flow_control_custom:
                tel.feature.flow_control_custom = True
            if self.settings._require_core:
                tel.feature.core = True
            tel.env.maybe_mp = _maybe_mp_process(backend)
        if not self.settings.label_disable:
            if self.notebook:
                run._label_probe_notebook(self.notebook)
            else:
                run._label_probe_main()
        for deprecated_feature, msg in self.deprecated_features_used.items():
            warning_message = f'`{deprecated_feature}` is deprecated. {msg}'
            deprecate(field_name=getattr(Deprecated, 'init__' + deprecated_feature), warning_message=warning_message, run=run)
        logger.info('updated telemetry')
        run._set_library(self._wl)
        run._set_backend(backend)
        run._set_reporter(self._reporter)
        run._set_teardown_hooks(self._teardown_hooks)
        backend._hack_set_run(run)
        assert backend.interface
        mailbox.enable_keepalive()
        backend.interface.publish_header()
        if not self.settings.disable_git:
            run._populate_git_info()
        run_result: Optional[pb.RunUpdateResult] = None
        if self.settings._offline:
            with telemetry.context(run=run) as tel:
                tel.feature.offline = True
            if self.settings.resume:
                wandb.termwarn(f'`resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id {run.id}.')
        error: Optional[wandb.errors.Error] = None
        timeout = self.settings.init_timeout
        logger.info(f'communicating run to backend with {timeout} second timeout')
        run_init_handle = backend.interface.deliver_run(run)
        result = run_init_handle.wait(timeout=timeout, on_progress=self._on_progress_init, cancel=True)
        if result:
            run_result = result.run_result
        if run_result is None:
            error_message = f'Run initialization has timed out after {timeout} sec. \nPlease refer to the documentation for additional information: {wburls.get('doc_start_err')}'
            error = CommError(error_message)
            run_init_handle._cancel()
        elif run_result.HasField('error'):
            error = ProtobufErrorHandler.to_exception(run_result.error)
        if error is not None:
            logger.error(f'encountered error: {error}')
            if not manager:
                backend.cleanup()
                self.teardown()
            raise error
        assert run_result is not None
        if not run_result.HasField('run'):
            raise Error("It appears that something have gone wrong during the program execution as an unexpected missing field was encountered. (run_result is missing the 'run' field)")
        if run_result.run.resumed:
            logger.info('run resumed')
            with telemetry.context(run=run) as tel:
                tel.feature.resumed = run_result.run.resumed
        run._set_run_obj(run_result.run)
        run._on_init()
        logger.info('starting run threads in backend')
        if manager:
            manager._inform_start(settings=self.settings.to_proto(), run_id=self.settings.run_id)
        assert backend.interface
        assert run._run_obj
        run_start_handle = backend.interface.deliver_run_start(run._run_obj)
        run_start_result = run_start_handle.wait(timeout=30)
        if run_start_result is None:
            run_start_handle.abandon()
        assert self._wl is not None
        self._wl._global_run_stack.append(run)
        self.run = run
        run._handle_launch_artifact_overrides()
        if self.settings.launch and self.settings.launch_config_path and os.path.exists(self.settings.launch_config_path):
            run._save(self.settings.launch_config_path)
        for k, v in self.init_artifact_config.items():
            run.config.update({k: v}, allow_val_change=True)
        job_artifact = run._launch_artifact_mapping.get(wandb.util.LAUNCH_JOB_ARTIFACT_SLOT_NAME)
        if job_artifact:
            run.use_artifact(job_artifact)
        self.backend = backend
        assert self._reporter
        self._reporter.set_context(run=run)
        run._on_start()
        logger.info('run started, returning control to user process')
        return run