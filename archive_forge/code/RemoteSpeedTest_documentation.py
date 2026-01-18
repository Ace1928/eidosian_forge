import argparse
import itertools
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

This example demonstrates the use of RemoteGraphicsView to improve performance in
applications with heavy load. It works by starting a second process to handle 
all graphics rendering, thus freeing up the main process to do its work.

In this example, the update() function is very expensive and is called frequently.
After update() generates a new set of data, it can either plot directly to a local
plot (bottom) or remotely via a RemoteGraphicsView (top), allowing speed comparison
between the two cases. IF you have a multi-core CPU, it should be obvious that the 
remote case is much faster.
