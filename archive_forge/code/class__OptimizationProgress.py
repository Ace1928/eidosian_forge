from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _OptimizationProgress(_BaseProgress):
    """Track optimization progress.

    Args:
        optimizer: Tracks optimizer progress.

    """
    optimizer: _OptimizerProgress = field(default_factory=_OptimizerProgress)

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed

    @override
    def reset(self) -> None:
        self.optimizer.reset()

    def reset_on_run(self) -> None:
        self.optimizer.reset_on_run()

    def reset_on_restart(self) -> None:
        self.optimizer.reset_on_restart()

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])