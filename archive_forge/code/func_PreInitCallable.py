from typing import Any, Callable, Optional
import wandb
def PreInitCallable(name: str, destination: Optional[Any]=None) -> Callable:

    def preinit_wrapper(*args: Any, **kwargs: Any) -> Any:
        raise wandb.Error(f'You must call wandb.init() before {name}()')
    preinit_wrapper.__name__ = str(name)
    if destination:
        preinit_wrapper.__wrapped__ = destination
        preinit_wrapper.__doc__ = destination.__doc__
    return preinit_wrapper