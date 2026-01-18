import os
from torch._C._autograd import _supported_activities, DeviceType, kineto_available
from torch._C._profiler import _ExperimentalConfig, ProfilerActivity, RecordScope
from torch.autograd.profiler import KinetoStepTracker, record_function
from torch.optim.optimizer import register_optimizer_step_post_hook
from .profiler import (
from . import itt
def _optimizer_post_hook(optimizer, args, kwargs):
    KinetoStepTracker.increment_step('Optimizer')