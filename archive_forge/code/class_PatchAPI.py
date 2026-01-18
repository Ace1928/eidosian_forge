import asyncio
import functools
import inspect
import logging
import sys
from typing import Any, Dict, Optional, Sequence, TypeVar
import wandb.sdk
import wandb.util
from wandb.sdk.lib import telemetry as wb_telemetry
from wandb.sdk.lib.timer import Timer
class PatchAPI:

    def __init__(self, name: str, symbols: Sequence[str], resolver: ArgumentResponseResolver) -> None:
        """Patches the API to log wandb Media or metrics."""
        self.name = name
        self._api = None
        self.original_methods: Dict[str, Any] = {}
        self.symbols = symbols
        self.resolver = resolver

    @property
    def set_api(self) -> Any:
        """Returns the API module."""
        lib_name = self.name.lower()
        if self._api is None:
            self._api = wandb.util.get_module(name=lib_name, required=f'To use the W&B {self.name} Autolog, you need to have the `{lib_name}` python package installed. Please install it with `pip install {lib_name}`.', lazy=False)
        return self._api

    def patch(self, run: 'wandb.sdk.wandb_run.Run') -> None:
        """Patches the API to log media or metrics to W&B."""
        for symbol in self.symbols:
            symbol_parts = symbol.split('.')
            original = functools.reduce(getattr, symbol_parts, self.set_api)

            def method_factory(original_method: Any):

                async def async_method(*args, **kwargs):
                    future = asyncio.Future()

                    async def callback(coro):
                        try:
                            result = await coro
                            loggable_dict = self.resolver(args, kwargs, result, timer.start_time, timer.elapsed)
                            if loggable_dict is not None:
                                run.log(loggable_dict)
                            future.set_result(result)
                        except Exception as e:
                            logger.warning(e)
                    with Timer() as timer:
                        coro = original_method(*args, **kwargs)
                        asyncio.ensure_future(callback(coro))
                    return await future

                def sync_method(*args, **kwargs):
                    with Timer() as timer:
                        result = original_method(*args, **kwargs)
                        try:
                            loggable_dict = self.resolver(args, kwargs, result, timer.start_time, timer.elapsed)
                            if loggable_dict is not None:
                                run.log(loggable_dict)
                        except Exception as e:
                            logger.warning(e)
                        return result
                if inspect.iscoroutinefunction(original_method):
                    return functools.wraps(original_method)(async_method)
                else:
                    return functools.wraps(original_method)(sync_method)
            self.original_methods[symbol] = original
            if len(symbol_parts) == 1:
                setattr(self.set_api, symbol_parts[0], method_factory(original))
            else:
                setattr(functools.reduce(getattr, symbol_parts[:-1], self.set_api), symbol_parts[-1], method_factory(original))

    def unpatch(self) -> None:
        """Unpatches the API."""
        for symbol, original in self.original_methods.items():
            symbol_parts = symbol.split('.')
            if len(symbol_parts) == 1:
                setattr(self.set_api, symbol_parts[0], original)
            else:
                setattr(functools.reduce(getattr, symbol_parts[:-1], self.set_api), symbol_parts[-1], original)