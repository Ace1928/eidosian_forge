import numpy as np
import time
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.ax import AxSearch
This example demonstrates the usage of AxSearch with Ray Tune.

It also checks that it is usable with a separate scheduler.

Requires the Ax library to be installed (`pip install ax-platform sqlalchemy`).
