from typing import Optional, Sequence, Union
import wandb
from wandb.errors import UnsupportedError
from wandb.sdk import wandb_run
from wandb.sdk.lib.wburls import wburls
class _Requires:
    """Internal feature class."""
    _features: Sequence[str]

    def __init__(self, features: Union[str, Sequence[str]]) -> None:
        self._features = tuple([features]) if isinstance(features, str) else tuple(features)

    def require_require(self) -> None:
        pass

    def _require_service(self) -> None:
        wandb.teardown = wandb._teardown
        wandb.attach = wandb._attach
        wandb_run.Run.detach = wandb_run.Run._detach

    def require_service(self) -> None:
        self._require_service()

    def apply(self) -> None:
        """Call require_* method for supported features."""
        last_message: str = ''
        for feature_item in self._features:
            full_feature = feature_item.split('@', 2)[0]
            feature = full_feature.split(':', 2)[0]
            func_str = 'require_{}'.format(feature.replace('-', '_'))
            func = getattr(self, func_str, None)
            if not func:
                last_message = f'require() unsupported requirement: {feature}'
                wandb.termwarn(last_message)
                continue
            func()
        if last_message:
            wandb.termerror(f'Supported wandb.require() features can be found at: {wburls.get('doc_require')}')
            raise UnsupportedError(last_message)