import ray
import os
from functools import wraps
import threading
Wrap public APIs with automatic ray.init.