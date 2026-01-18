from collections import defaultdict
import numpy as np
from moviepy.decorators import use_clip_fps_by_default
class FramesMatch:
    """
    
    Parameters
    -----------

    t1
      Starting time

    t2
      End time

    d_min
      Lower bound on the distance between the first and last frames

    d_max
      Upper bound on the distance between the first and last frames

    """

    def __init__(self, t1, t2, d_min, d_max):
        self.t1 = t1
        self.t2 = t2
        self.d_min = d_min
        self.d_max = d_max
        self.time_span = t2 - t1

    def __str__(self):
        return '(%.04f, %.04f, %.04f, %.04f)' % (self.t1, self.t2, self.d_min, self.d_max)

    def __repr__(self):
        return '(%.04f, %.04f, %.04f, %.04f)' % (self.t1, self.t2, self.d_min, self.d_max)

    def __iter__(self):
        return iter((self.t1, self.t2, self.d_min, self.d_max))