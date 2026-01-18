import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
class _Manager:
    _token: _ManagerToken
    _atexit_lambda: Optional[Callable[[], None]]
    _hooks: Optional[ExitHooks]
    _settings: 'Settings'
    _service: 'service._Service'

    def _service_connect(self) -> None:
        port = self._token.port
        svc_iface = self._get_service_interface()
        try:
            svc_iface._svc_connect(port=port)
        except ConnectionRefusedError as e:
            if not psutil.pid_exists(self._token.pid):
                message = 'Connection to wandb service failed since the process is not available. '
            else:
                message = f'Connection to wandb service failed: {e}. '
            raise ManagerConnectionRefusedError(message)
        except Exception as e:
            raise ManagerConnectionError(f'Connection to wandb service failed: {e}')

    def __init__(self, settings: 'Settings') -> None:
        from wandb.sdk.service import service
        self._settings = settings
        self._atexit_lambda = None
        self._hooks = None
        self._service = service._Service(settings=self._settings)
        token = _ManagerToken.from_environment()
        if not token:
            self._service.start()
            host = 'localhost'
            transport = 'tcp'
            port = self._service.sock_port
            assert port
            token = _ManagerToken.from_params(transport=transport, host=host, port=port)
            token.set_environment()
            self._atexit_setup()
        self._token = token
        try:
            self._service_connect()
        except ManagerConnectionError as e:
            wandb._sentry.reraise(e)

    def _atexit_setup(self) -> None:
        self._atexit_lambda = lambda: self._atexit_teardown()
        self._hooks = ExitHooks()
        self._hooks.hook()
        atexit.register(self._atexit_lambda)

    def _atexit_teardown(self) -> None:
        trigger.call('on_finished')
        exit_code = self._hooks.exit_code if self._hooks else 0
        self._teardown(exit_code)

    def _teardown(self, exit_code: int) -> None:
        unregister_all_post_import_hooks()
        if self._atexit_lambda:
            atexit.unregister(self._atexit_lambda)
            self._atexit_lambda = None
        try:
            self._inform_teardown(exit_code)
            result = self._service.join()
            if result and (not self._settings._notebook):
                os._exit(result)
        except Exception as e:
            wandb.termlog(f'While tearing down the service manager. The following error has occurred: {e}', repeat=False)
        finally:
            self._token.reset_environment()

    def _get_service(self) -> 'service._Service':
        return self._service

    def _get_service_interface(self) -> 'ServiceInterface':
        assert self._service
        svc_iface = self._service.service_interface
        assert svc_iface
        return svc_iface

    def _inform_init(self, settings: 'wandb_settings_pb2.Settings', run_id: str) -> None:
        svc_iface = self._get_service_interface()
        svc_iface._svc_inform_init(settings=settings, run_id=run_id)

    def _inform_start(self, settings: 'wandb_settings_pb2.Settings', run_id: str) -> None:
        svc_iface = self._get_service_interface()
        svc_iface._svc_inform_start(settings=settings, run_id=run_id)

    def _inform_attach(self, attach_id: str) -> Optional[Dict[str, Any]]:
        svc_iface = self._get_service_interface()
        try:
            response = svc_iface._svc_inform_attach(attach_id=attach_id)
        except Exception:
            return None
        return response.settings

    def _inform_finish(self, run_id: Optional[str]=None) -> None:
        svc_iface = self._get_service_interface()
        svc_iface._svc_inform_finish(run_id=run_id)

    def _inform_teardown(self, exit_code: int) -> None:
        svc_iface = self._get_service_interface()
        svc_iface._svc_inform_teardown(exit_code)