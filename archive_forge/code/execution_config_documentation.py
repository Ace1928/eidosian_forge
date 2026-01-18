from dataclasses import dataclass
from typing import Optional
from pennylane.workflow import SUPPORTED_INTERFACES
from pennylane.gradients import SUPPORTED_GRADIENT_KWARGS

        Validate the configured execution options.

        Note that this hook is automatically called after init via the dataclass integration.
        