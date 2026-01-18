from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _OptimizerProgress(_BaseProgress):
    """Track optimizer progress.

    Args:
        step: Tracks ``optimizer.step`` calls.
        zero_grad: Tracks ``optimizer.zero_grad`` calls.

    """
    step: _Progress = field(default_factory=lambda: _Progress.from_defaults(_ReadyCompletedTracker))
    zero_grad: _Progress = field(default_factory=lambda: _Progress.from_defaults(_StartedTracker))

    @override
    def reset(self) -> None:
        self.step.reset()
        self.zero_grad.reset()

    def reset_on_run(self) -> None:
        self.step.reset_on_run()
        self.zero_grad.reset_on_run()

    def reset_on_restart(self) -> None:
        self.step.reset_on_restart()
        self.zero_grad.reset_on_restart()

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.step.load_state_dict(state_dict['step'])
        self.zero_grad.load_state_dict(state_dict['zero_grad'])