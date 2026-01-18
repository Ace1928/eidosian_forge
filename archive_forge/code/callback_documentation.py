import logging
import os
from typing import Collection, List, Optional, Type, Union, TYPE_CHECKING
from ray.tune.callback import Callback, CallbackList
from ray.tune.logger import (
Create default callbacks for `Tuner.fit()`.

    This function takes a list of existing callbacks and adds default
    callbacks to it.

    Specifically, three kinds of callbacks will be added:

    1. Loggers. Ray Tune's experiment analysis relies on CSV and JSON logging.
    2. Syncer. Ray Tune synchronizes logs and checkpoint between workers and
       the head node.
    2. Trial progress reporter. For reporting intermediate progress, like trial
       results, Ray Tune uses a callback.

    These callbacks will only be added if they don't already exist, i.e. if
    they haven't been passed (and configured) by the user. A notable case
    is when a Logger is passed, which is not a CSV or JSON logger - then
    a CSV and JSON logger will still be created.

    Lastly, this function will ensure that the Syncer callback comes after all
    Logger callbacks, to ensure that the most up-to-date logs and checkpoints
    are synced across nodes.

    