import logging
from typing import TYPE_CHECKING
import ray
from .backpressure_policy import BackpressurePolicy
A backpressure policy that caps the concurrency of each operator.

    The concurrency cap limits the number of concurrently running tasks.
    It will be set to an intial value, and will ramp up exponentially.

    The concrete stategy is as follows:
    - Each PhysicalOperator is assigned an initial concurrency cap.
    - An PhysicalOperator can run new tasks if the number of running tasks is less
      than the cap.
    - When the number of finished tasks reaches a threshold, the concurrency cap will
      increase.
    